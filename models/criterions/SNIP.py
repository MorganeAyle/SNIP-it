import os

import torch
import torch.nn.functional as F

from tqdm import tqdm

from models.criterions.General import General
from utils.constants import RESULTS_DIR, OUTPUT_DIR, SNIP_BATCH_ITERATIONS
from utils.attacks_utils import construct_adversarial_examples


class SNIP(General):
    """
    Our interpretation/implementation of SNIP from the paper:
    SNIP: Single-shot Network Pruning based on Connection Sensitivity
    https://arxiv.org/abs/1810.02340
    Additionally, edited to use elasticity as a criterion instead of sensitivity, which we describe and justify in our paper:
    https://arxiv.org/abs/2006.00896
    """

    def __init__(self, *args, **kwargs):
        super(SNIP, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage, train_loader=None, manager=None, ood_loader=None, local=False, iterations=5, **kwargs):

        self.iterations = iterations

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

            # how much we wanna prune
            length_nonzero = float(self.model.mask[name].flatten().shape[0])

            cutoff = (self.model.mask[name] == 0).sum().item()

            # print("pruning", name, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
        self.model.apply_weight_mask()
        # print("final percentage after snip:", self.model.pruned_percentage)
        # self.cut_lonely_connections()

    def get_weight_saliencies(self, train_loader, ood_loader=None):

        net = self.model.eval()

        iterations = self.iterations

        # accumalate gradients of multiple batches
        net.zero_grad()
        loss_sum = torch.zeros([1]).to(self.device)
        for i, (x, y) in enumerate(train_loader):

            if i == iterations: break

            inputs = x.to(self.model.device)
            targets = y.to(self.model.device)

            outputs = net.forward(inputs)

            loss = F.nll_loss(outputs, targets) / iterations

            loss.backward()
            loss_sum += loss.item()

            # import torchvision
            # torchvision.utils.save_image(x, '/nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/test.png',
                                         # normalize=True)

        # get elasticities
        grads_abs = {}
        self.grads_abs = {}
        for name, layer in net.named_modules():
            if "Norm" in str(layer): continue
            if name + ".weight" in self.model.mask:
                grads_abs[name + ".weight"] = torch.abs(
                    layer.weight.grad * (layer.weight.data / (1e-8 + loss_sum.item())))
        all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])
        norm_factor = 1
        log10 = all_scores.sort().values.log10()
        all_scores.div_(norm_factor)

        self.model = self.model.train()

        return all_scores, grads_abs, log10, norm_factor
