import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.attacks_utils import construct_adversarial_examples
from utils.metrics import calculate_aupr, calculate_auroc, calculate_brier_score


def calculate_ece(all_preds, all_correct, n_bins=10):
    hist, bin_edges = np.histogram(all_preds, n_bins)
    per_bin_acc = np.zeros(n_bins)
    per_bin_conf = np.zeros(n_bins)
    N = len(all_preds)

    for i, pred in enumerate(all_preds):
        for j in range(n_bins):
            lower = bin_edges[j]
            upper = bin_edges[j+1]
            if j == n_bins:
                if lower <= pred <= upper:
                    per_bin_conf[j] += pred / hist[j]
                    per_bin_acc[j] += all_correct[i] / hist[j]
            else:
                if lower <= pred < upper:
                    per_bin_conf[j] += pred / hist[j]
                    per_bin_acc[j] += all_correct[i] / hist[j]
                    break
    ece = np.sum((hist / N) * np.absolute(per_bin_acc - per_bin_conf))

    return ece


class InEvaluation:

    """
    Performs evaluation on test data
    """

    def __init__(self, test_loader, model, device, ensemble=None, **kwargs):
        self.device = device
        self.model = model
        self.test_loader = test_loader
        self.ensemble = ensemble

    def evaluate(self, **kwargs):
        true_labels = np.zeros(0)
        all_preds = np.zeros(0)
        all_correct = np.zeros(0)
        conf_true_labels = np.zeros(0)
        brier_scores = []
        entropies = np.zeros(0)
        acc = []
        nll = []

        with torch.no_grad():
            for batch_num, batch in enumerate(self.test_loader):
                x, y = batch
                x = x.to(self.device)

                if not self.ensemble:
                    out = self.model(x)
                else:
                    out = 0
                    for model in self.ensemble:
                        out += model(x)
                    out /= len(self.ensemble)
                # Logits to probability distribution
                probs = F.softmax(out, dim=-1)
                # Maximum softmax probability
                preds, indices = torch.max(probs, dim=-1)
                # Label predictions
                label_preds = probs.argmax(dim=-1, keepdim=True).view_as(y)
                # Compute accuracy
                corrects = y.eq(label_preds.cpu())
                correct = corrects.sum().item()
                acc.append(correct / out.shape[0])

                all_correct = np.concatenate((all_correct, corrects.cpu().numpy()))

                # Compute entropy
                entropy = Categorical(probs).entropy().squeeze()
                entropies = np.concatenate((entropies, entropy.cpu().numpy()))

                # Compute brier score
                brier_scores.append(calculate_brier_score(probs, y))

                # Compute NLL
                nll.append(-np.mean(np.log(preds.cpu().numpy())))

                true_labels = np.concatenate((true_labels, np.ones(len(x))))
                all_preds = np.concatenate((all_preds, preds.cpu().reshape((-1))))
                conf_true_labels = np.concatenate((conf_true_labels, torch.isclose(y.cpu(), indices.cpu()).numpy().astype(float).reshape(-1)))

        conf_auroc = calculate_auroc(conf_true_labels, all_preds)
        conf_aupr = calculate_aupr(conf_true_labels, all_preds)
        brier_score = np.mean(np.array(brier_scores))
        ece = calculate_ece(all_preds, all_correct)

        return {'conf_auroc': conf_auroc,
                'conf_aupr': conf_aupr,
                'brier_score': brier_score,
                'entropy': np.mean(entropies),
                'test_acc': np.mean(acc),
                'nll': np.mean(nll),
                'ece': ece,
                }, true_labels, all_preds, entropies
