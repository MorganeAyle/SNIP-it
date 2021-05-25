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

    def __init__(self, test_loader, ood_loader, model, device, arguments, ood_dataset, **kwargs):
        self.args = arguments
        self.device = device
        self.model = model
        self.test_loader = test_loader
        self.ood_loader = ood_loader
        self.ood_dataset = ood_dataset

    def evaluate(self, **kwargs):
        true_labels = np.zeros(0)
        all_preds = np.zeros(0)
        conf_true_labels = np.zeros(0)
        brier_scores = []
        entropies = []
        ood_entropies = []

        with torch.no_grad():
            for batch_num, batch in enumerate(self.test_loader):
                x, y = batch

                x = x.to(self.device)

                out = self.model(x)
                probs = F.softmax(out, dim=-1)
                preds, indices = torch.max(probs, dim=-1)

                entropy = Categorical(probs).entropy().squeeze().mean()
                entropies.append(entropy.cpu().numpy())

                brier_scores.append(calculate_brier_score(probs, y))

                true_labels = np.concatenate((true_labels, np.ones(len(x))))
                all_preds = np.concatenate((all_preds, preds.cpu().reshape((-1))))
                conf_true_labels = np.concatenate((conf_true_labels, torch.isclose(y.cpu(), indices.cpu()).numpy().astype(float).reshape(-1)))

        conf_auroc = calculate_auroc(conf_true_labels, all_preds)
        conf_aupr = calculate_aupr(conf_true_labels, all_preds)
        brier_score = np.mean(np.array(brier_scores))

        with torch.no_grad():
            for batch_num, batch in enumerate(self.ood_loader):
                x, y = batch

                x = x.float().to(self.device)

                out = self.model(x)
                probs = F.softmax(out, dim=-1)
                preds, _ = torch.max(probs, dim=-1)

                entropy = Categorical(probs).entropy().squeeze().mean()
                ood_entropies.append(entropy.cpu().numpy())

                true_labels = np.concatenate((true_labels, np.zeros(len(x))))
                all_preds = np.concatenate((all_preds, preds.cpu().reshape((-1))))

        auroc = calculate_auroc(true_labels, all_preds)
        aupr = calculate_aupr(true_labels, all_preds)

        auroc_name = f'auroc_{self.ood_dataset}'
        aupr_name = f'aupr_{self.ood_dataset}'
        entropy_name = f'entropy_{self.ood_dataset}'

        return {'conf_auroc': conf_auroc,
                'conf_aupr': conf_aupr,
                'brier_score': brier_score,
                'entropy': np.mean(entropies),
                auroc_name: auroc,
                aupr_name: aupr,
                entropy_name: np.mean(ood_entropies)}
