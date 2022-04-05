import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.attacks_utils import construct_adversarial_examples
from utils.metrics import calculate_aupr, calculate_auroc, calculate_brier_score


class OODEvaluation:

    """
    Performs evaluation to adversarial attacks
    """

    def __init__(self, ood_loader, model, device, ood_dataset, ensemble=None, **kwargs):
        self.device = device
        self.model = model
        self.ood_loader = ood_loader
        self.ood_dataset = ood_dataset
        self.ensemble = ensemble

    def evaluate(self, true_labels, all_preds, entropies, **kwargs):
        ood_entropies = np.zeros(0)

        with torch.no_grad():
            for batch_num, batch in enumerate(self.ood_loader):
                x, y = batch
                x = x.float().to(self.device)

                if not self.ensemble:
                    out = self.model(x)
                else:
                    out = 0
                    for model in self.ensemble:
                        out += model(x)
                    out /= len(self.ensemble)
                probs = F.softmax(out, dim=-1)
                preds, _ = torch.max(probs, dim=-1)

                entropy = Categorical(probs).entropy().squeeze()
                entropies = np.concatenate((entropies, entropy.detach().cpu().numpy()))
                ood_entropies = np.concatenate((ood_entropies, entropy.cpu().numpy()))

                true_labels = np.concatenate((true_labels, np.zeros(len(x))))
                all_preds = np.concatenate((all_preds, preds.cpu().reshape((-1))))

        auroc = calculate_auroc(true_labels, all_preds)
        aupr = calculate_aupr(true_labels, all_preds)

        auroc_entropy = calculate_auroc(1 - true_labels, entropies)
        aupr_entropy = calculate_aupr(1 - true_labels, entropies)

        auroc_name = f'auroc_{self.ood_dataset}'
        aupr_name = f'aupr_{self.ood_dataset}'
        auroc_ent_name = f'auroc_entropy_{self.ood_dataset}'
        aupr_ent_name = f'aupr_entropy_{self.ood_dataset}'
        entropy_name = f'entropy_{self.ood_dataset}'

        return {auroc_name: auroc,
                aupr_name: aupr,
                entropy_name: np.mean(ood_entropies),
                auroc_ent_name: auroc_entropy,
                aupr_ent_name: aupr_entropy
                }
