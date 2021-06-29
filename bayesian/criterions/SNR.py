import os

import torch
import torch.nn.functional as F

from models.criterions.General import General
from utils.constants import RESULTS_DIR, OUTPUT_DIR, SNIP_BATCH_ITERATIONS
from utils.attacks_utils import construct_adversarial_examples
import bayesian_utils as butils
import metrics


class SNR(General):
    """
    Our interpretation/implementation of SNIP from the paper:
    SNIP: Single-shot Network Pruning based on Connection Sensitivity
    https://arxiv.org/abs/1810.02340
    Additionally, edited to use elasticity as a criterion instead of sensitivity, which we describe and justify in our paper:
    https://arxiv.org/abs/2006.00896
    """

    def __init__(self, *args, **kwargs):
        super(SNR, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage, train_loader=None, manager=None, ood_loader=None, local=False, **kwargs):

        all_scores, grads_abs, log10, norm_factor = self.get_weight_saliencies(train_loader, ood_loader)

        self.handle_pruning(all_scores, grads_abs, log10, manager, norm_factor, percentage, local)

    def handle_pruning(self, all_scores, grads_abs, log10, manager, norm_factor, percentage, local):
        from utils.constants import RESULTS_DIR
        if manager is not None:
            manager.save_python_obj(all_scores.cpu().numpy(),
                                    os.path.join(RESULTS_DIR, manager.stamp, OUTPUT_DIR, f"scores"))

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

            print(self.model.mask[name].sum().item())
            self.model.mask[name] = ((grad / norm_factor) > acceptable_score).__and__(
                self.model.mask[name].bool()).float().to(self.device)

            # how much we wanna prune
            length_nonzero = float(self.model.mask[name].flatten().shape[0])

            cutoff = (self.model.mask[name] == 0).sum().item()

            print("pruning", name, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)

        for name, module in self.model.named_modules():
            if name in self.model.mask:
                module.mask = self.model.mask[name]

        # self.model.apply_weight_mask()
        print("final percentage after snip:", self.model.pruned_percentage)
        # self.cut_lonely_connections()

    def get_weight_saliencies(self, train_loader, ood_loader=None):

        net = self.model.eval()

        print(self.model.mask.keys())

        # get elasticities
        grads_abs = {}
        for name, layer in net.named_modules():
            if 'conv' in name or 'fc' in name:
                grads_abs[name] = torch.abs(
                    layer.W_mu.data) / torch.log1p(torch.exp(layer.W_rho.data))
        all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])
        norm_factor = 1
        log10 = all_scores.sort().values.log10()
        all_scores.div_(norm_factor)

        self.model = self.model.train()

        return all_scores, grads_abs, log10, norm_factor
