from sklearn import metrics
import numpy as np


def calculate_auroc(correct, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(correct, predictions)
    auroc = metrics.auc(fpr, tpr)
    return auroc


def calculate_aupr(correct, predictions):
    aupr = metrics.average_precision_score(correct, predictions)
    return aupr


def calculate_brier_score(probs, labels):
    batch_size = probs.shape[0]

    indices = np.arange(batch_size)
    probs[indices, labels] -= 1
    brier_score = probs.norm(dim=-1).mean()
    return brier_score.cpu().detach().numpy()
