import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.attacks_utils import construct_adversarial_examples
from utils.metrics import calculate_aupr, calculate_auroc, calculate_brier_score


class AdversarialEvaluation:

    """
    Performs evaluation to adversarial attacks
    """

    def __init__(self, attack, test_loader, model, device, arguments, **kwargs):
        self.args = arguments
        self.method = attack
        self.device = device
        self.model = model
        self.test_loader = test_loader

    def evaluate(self, targeted=False, exclude_wrong_predictions=False, epsilon=6, **kwargs):

        method = self.method

        model = self.model.eval().to(self.device)

        success_rates = []
        true_labels = np.zeros(0)
        all_preds = np.zeros(0)
        entropies = []
        ood_entropies = []
        for im, crit in self.test_loader:

            # probs for indistribution
            x, y = im, crit
            x = x.to(self.device)

            out = self.model(x)
            probs = F.softmax(out, dim=-1)
            preds, indices = torch.max(probs, dim=-1)

            entropy = Categorical(probs).entropy().squeeze().mean()
            entropies.append(entropy.detach().cpu().numpy())

            true_labels = np.concatenate((true_labels, np.ones(len(x))))
            all_preds = np.concatenate((all_preds, preds.detach().cpu().reshape((-1))))

            adv_results, predictions = construct_adversarial_examples(im, crit, method, model, self.device, epsilon, exclude_wrong_predictions, targeted)
            _, advs, success = adv_results

            attack_success = (success.float().sum() / len(success)).item()
            adv_equality = (model.forward(advs).argmax(dim=-1) == predictions)
            predicted_same_as_model = (adv_equality.float().sum() / len(success)).item()

            advs = advs.cpu()

            success_rates.append(attack_success)

            print("EPSILON", epsilon,
                "Successes attack", attack_success,
                "same prediction as model", predicted_same_as_model,
                "bounds adver", advs.min().item(), advs.max().item(),
                "norm", np.mean([torch.norm(eps_adv_ - im_, p=2).item() for eps_adv_, im_ in zip(advs, im)]))

            # probs for outofdistribution
            x = advs.to(self.device)

            out = self.model(x)
            probs = F.softmax(out, dim=-1)
            preds, indices = torch.max(probs, dim=-1)

            entropy = Categorical(probs).entropy().squeeze().mean()
            ood_entropies.append(entropy.detach().cpu().numpy())

            true_labels = np.concatenate((true_labels, np.zeros(len(x))))
            all_preds = np.concatenate((all_preds, preds.detach().cpu().reshape((-1))))

        success_name = f'success_rate_{self.method}_{epsilon}'

        auroc = calculate_auroc(true_labels, all_preds)
        aupr = calculate_aupr(true_labels, all_preds)

        auroc_name = f'auroc_{self.method}_{epsilon}'
        aupr_name = f'aupr_{self.method}_{epsilon}'
        entropy_name = f'entropy_{self.method}_{epsilon}'

        return {success_name: np.mean(success_rates),
                auroc_name: auroc,
                aupr_name: aupr,
                entropy_name: np.mean(ood_entropies)
                }
