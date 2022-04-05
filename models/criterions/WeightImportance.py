import os

import torch
import torch.nn.functional as F

from tqdm import tqdm

from models.criterions.General import General
from utils.constants import RESULTS_DIR, OUTPUT_DIR, SNIP_BATCH_ITERATIONS
from utils.attacks_utils import construct_adversarial_examples


class WeightImportance(General):
    """
    Our interpretation/implementation of SNIP from the paper:
    SNIP: Single-shot Network Pruning based on Connection Sensitivity
    https://arxiv.org/abs/1810.02340
    Additionally, edited to use elasticity as a criterion instead of sensitivity, which we describe and justify in our paper:
    https://arxiv.org/abs/2006.00896
    """

    def __init__(self, limit=0.0, steps=5, orig_scores=False, *args, **kwargs):
        super(WeightImportance, self).__init__(*args, **kwargs)
        # always smaller than limit, steps+1 elements (including limit)
        # self.steps = [limit - (limit - lower_limit) * (0.5 ** i) for i in range(steps - 1)] + [limit]
        self.orig_scores = orig_scores

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage, train_loader=None, manager=None, ood_loader=None, local=False, iterations=5, **kwargs):
        self.iterations = iterations
        self.grads_abs = None

        all_scores, grads_abs, log10, norm_factor = self.get_weight_saliencies(train_loader, ood_loader)

        self.handle_pruning(all_scores, grads_abs, log10, manager, norm_factor, percentage, local)

    def handle_pruning(self, all_scores, grads_abs, log10, manager, norm_factor, percentage, local):
        from utils.constants import RESULTS_DIR
        if manager is not None:
            manager.save_python_obj(all_scores.cpu().detach().numpy(),
                                    os.path.join(RESULTS_DIR, manager.stamp, OUTPUT_DIR, f"scores"))

        # print(local)

        if not local:
            # don't prune more or less than possible
            num_params_to_keep = int(len(all_scores) * (1 - percentage))
            if num_params_to_keep < 1:
                num_params_to_keep += 1
            elif num_params_to_keep > len(all_scores):
                num_params_to_keep = len(all_scores)

            # threshold
            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

        # prune
        self.grads_abs = {}
        for name, grad in grads_abs.items():

            if local:
                # don't prune more or less than possible
                num_params_to_keep = int(len(torch.flatten(grad)) * (1 - percentage))
                if num_params_to_keep < 1:
                    num_params_to_keep += 1
                elif num_params_to_keep > len(torch.flatten(grad)):
                    num_params_to_keep = len(torch.flatten(grad))

                # threshold
                threshold, _ = torch.topk(torch.flatten(grad), num_params_to_keep, sorted=True)
                acceptable_score = threshold[-1]

            # print(self.model.mask[name].sum().item())
            self.model.mask[name] = ((grad / norm_factor) > acceptable_score).__and__(
                self.model.mask[name].bool()).float().to(self.device)

            self.grads_abs[name] = self.model.mask[name] * (grad / norm_factor)
            # self.grads_abs[name] = self.model.mask[name]

        self.model.apply_weight_mask()

    def get_weight_saliencies(self, train_loader, ood_loader=None):

        net = self.model.eval()

        # iterations = self.iterations
        iterations = len(train_loader)

        self.scores = None

        # accumalate gradients of multiple batches
        net.zero_grad()
        loss_sum = torch.zeros([1]).to(self.device)
        batches = []
        for i, (x, y) in enumerate(tqdm(train_loader)):
            batches.append((x, y))

            if i == iterations: break

            inputs = x.to(self.model.device)
            targets = y.to(self.model.device)

            outputs = net.forward(inputs)

            # loss = (outputs.mean(1) - torch.logsumexp(outputs, dim=1)).mean() / iterations

            loss = F.nll_loss(outputs, targets) / iterations

            loss.backward()
            loss_sum += loss.item()

            if self.orig_scores:
                if self.scores is None:
                    self.scores = {}
                    for name, layer in net.named_modules():
                        if "Norm" in str(layer): continue
                        if name + ".weight" in self.model.mask:
                            self.scores[name + ".weight"] = torch.unsqueeze(layer.weight.grad * (layer.weight.data / (1e-8 + loss_sum.item())), 0).cpu()
                else:
                    for name, layer in net.named_modules():
                        if "Norm" in str(layer): continue
                        if name + ".weight" in self.model.mask:
                            curr_scores = torch.unsqueeze(layer.weight.grad * (layer.weight.data / (1e-8 + loss_sum.item())), 0).cpu()
                            self.scores[name + ".weight"] = torch.cat((self.scores[name + ".weight"], curr_scores))

        self.scores_mean = {}
        self.scores_std = {}

        if self.orig_scores:
            for name, layer in net.named_modules():
                if "Norm" in str(layer): continue
                if name + ".weight" in self.model.mask:
                    self.scores_mean[name + ".weight"] = torch.mean(self.scores[name + ".weight"], dim=0)
                    self.scores_std[name + ".weight"] = torch.std(self.scores[name + ".weight"], dim=0, unbiased=True)

        # get elasticities
        grads_abs = {}
        self.grads_abs = {}
        for name, layer in net.named_modules():
            if "Norm" in str(layer): continue
            if name + ".weight" in self.model.mask:
                grads_abs[name + ".weight"] = layer.weight.grad * (layer.weight.data / (1e-8 + loss_sum.item()))

        # if self.orig_scores:
        #     self.scores_mean = grads_abs
        #     self.scores_std = {name: torch.zeros_like(val) for name, val in self.scores_mean.items()}
        #     for i, (x, y) in enumerate(tqdm(batches)):
        #
        #         if i == iterations: break
        #
        #         inputs = x.to(self.model.device)
        #         targets = y.to(self.model.device)
        #
        #         outputs = net.forward(inputs)
        #
        #         loss = F.nll_loss(outputs, targets) / iterations
        #
        #         loss.backward()
        #
        #         for name, layer in net.named_modules():
        #             if "Norm" in str(layer): continue
        #             if name + ".weight" in self.model.mask:
        #                 curr_scores = layer.weight.grad * (layer.weight.data / (1e-8 + loss_sum.item()))
        #                 self.scores_std[name + ".weight"] += (curr_scores - grads_abs[name + ".weight"]) ** 2
        #
        #     for name, val in self.scores_std.items():
        #         self.scores_std[name] = torch.sqrt(val / (len(train_loader) - 1))
        #
        all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])
        norm_factor = 1
        log10 = all_scores.sort().values.log10()
        all_scores.div_(norm_factor)

        self.model = self.model.train()
        # self.grads_abs = grads_abs

        return all_scores, grads_abs, log10, norm_factor
