import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.attacks_utils import construct_adversarial_examples
from utils.metrics import calculate_aupr, calculate_auroc, calculate_brier_score


class OODEvaluationLRC:

    """
    Performs evaluation to adversarial attacks
    """

    def __init__(self, test_loader, ood_loader, models, device, arguments, ood_dataset, **kwargs):
        self.args = arguments
        self.device = device
        self.models = models
        self.test_loader = test_loader
        self.ood_loader = ood_loader
        self.ood_dataset = ood_dataset

    def evaluate(self, **kwargs):
        true_labels = np.zeros(0)
        all_preds = np.zeros(0)

        with torch.no_grad():
            for batch_num, batch in enumerate(self.test_loader):
                x, y = batch

                x = x.to(self.device)

                for i, model in enumerate(self.models):
                    out = model(x)
                    _, indices = torch.max(out, dim=-1)

                    if i == 0:
                        all_indices = torch.unsqueeze(indices.cpu(), 1).numpy()
                    else:
                        all_indices = np.concatenate((all_indices, torch.unsqueeze(indices.cpu(), 1).numpy()), 1)

                lrc = []
                for labels in all_indices:
                    num_labels = len(np.unique(labels))
                    lrc.append(num_labels / len(self.models))

                true_labels = np.concatenate((true_labels, np.zeros(len(x))))
                all_preds = np.concatenate((all_preds, np.array(lrc)))

        with torch.no_grad():
            for batch_num, batch in enumerate(self.ood_loader):
                x, y = batch

                x = x.float().to(self.device)

                for i, model in enumerate(self.models):
                    out = model(x)
                    _, indices = torch.max(out, dim=-1)

                    if i == 0:
                        all_indices = torch.unsqueeze(indices.cpu(), 1).numpy()
                    else:
                        all_indices = np.concatenate((all_indices, torch.unsqueeze(indices.cpu(), 1).numpy()), 1)

                lrc = []
                for labels in all_indices:
                    num_labels = len(np.unique(labels))
                    lrc.append(num_labels / len(self.models))

                true_labels = np.concatenate((true_labels, np.ones(len(x))))
                all_preds = np.concatenate((all_preds, np.array(lrc)))

        auroc = calculate_auroc(true_labels, all_preds)
        aupr = calculate_aupr(true_labels, all_preds)

        auroc_name = f'auroc_{self.ood_dataset}'
        aupr_name = f'aupr_{self.ood_dataset}'

        return {
                auroc_name: auroc,
                aupr_name: aupr,
                }
