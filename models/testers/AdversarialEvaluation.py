import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.attacks_utils import construct_adversarial_examples
from utils.metrics import calculate_aupr, calculate_auroc

from torchvision import transforms

from copy import deepcopy


class AdversarialEvaluation:

    """
    Performs evaluation to adversarial attacks
    """

    def __init__(self, attack, test_loader, model, device, mean, std, ensemble=None, **kwargs):
        self.method = attack
        self.device = device
        self.model = model
        self.test_loader = test_loader
        self.ensemble = ensemble
        self.mean = mean
        self.std = std
        self.transform = transforms.Normalize(mean, std)

    def one_model_evaluation(self, model, epsilon, exclude_wrong_predictions, targeted, true_labels, all_preds, entropies):
        adv_acc = []
        ood_entropies = np.zeros(0)

        for im, crit in self.test_loader:
            adv_results, predictions = construct_adversarial_examples(im, crit, self.method, model, self.device,
                                                                      epsilon, exclude_wrong_predictions, targeted)
            _, advs, _ = adv_results

            advs = advs.cpu()
            new_x = []
            for image in advs:
                if image.dim() > 3:
                    image = self.transform(image.squeeze(0)).unsqueeze(0)
                else:
                    image = self.transform(image).unsqueeze(0)
                new_x.append(image)
            advs = torch.cat(new_x)

            advs = advs.to(self.device)

            adv_acc.append((model.forward(advs).argmax(dim=-1).cpu().flatten() == crit).float().sum().numpy() / len(im))

            x = advs

            out = model(x)
            probs = F.softmax(out, dim=-1)
            preds, indices = torch.max(probs, dim=-1)

            entropy = Categorical(probs).entropy().squeeze()
            ood_entropies = np.concatenate((ood_entropies, entropy.detach().cpu().numpy()))
            entropies = np.concatenate((entropies, entropy.detach().cpu().numpy()))

            true_labels = np.concatenate((true_labels, np.zeros(len(x))))
            all_preds = np.concatenate((all_preds, preds.detach().cpu().reshape((-1))))

        auroc = calculate_auroc(true_labels, all_preds)
        aupr = calculate_aupr(true_labels, all_preds)

        auroc_entropy = calculate_auroc(1 - true_labels, entropies)
        aupr_entropy = calculate_aupr(1 - true_labels, entropies)

        return np.mean(adv_acc), auroc, aupr, auroc_entropy, aupr_entropy, np.mean(ood_entropies)

    def evaluate(self, true_labels, all_preds, entropies, targeted=False, exclude_wrong_predictions=False, epsilon=6, **kwargs):

        if not self.ensemble:
            adv_acc, auroc, aupr, auroc_entropy, aupr_entropy, ood_entropy = \
                self.one_model_evaluation(self.model, epsilon, exclude_wrong_predictions, targeted, deepcopy(true_labels),
                                          deepcopy(all_preds), deepcopy(entropies))
        else:
            adv_acc, auroc, aupr, auroc_entropy, aupr_entropy, ood_entropy = [], [], [], [], [], []
            for model in self.ensemble:
                _adv_acc, _auroc, _aupr, _auroc_entropy, _aupr_entropy, _ood_entropy = \
                    self.one_model_evaluation(model, epsilon, exclude_wrong_predictions, targeted,
                                              deepcopy(true_labels), deepcopy(all_preds), deepcopy(entropies))
                adv_acc.append(_adv_acc)
                auroc.append(_auroc)
                aupr.append(_aupr)
                auroc_entropy.append(_auroc_entropy)
                aupr_entropy.append(_aupr_entropy)
                ood_entropy.append(_ood_entropy)
            adv_acc = np.mean(adv_acc)
            auroc = np.mean(auroc)
            aupr = np.mean(aupr)
            auroc_entropy = np.mean(auroc_entropy)
            aupr_entropy = np.mean(aupr_entropy)
            ood_entropy = np.mean(ood_entropy)

        adv_name = f'adv_acc_{self.method}_{epsilon}'

        auroc_name = f'auroc_{self.method}_{epsilon}'
        aupr_name = f'aupr_{self.method}_{epsilon}'
        auroc_ent_name = f'auroc_entropy_{self.method}_{epsilon}'
        aupr_ent_name = f'aupr_entropy_{self.method}_{epsilon}'
        entropy_name = f'entropy_{self.method}_{epsilon}'

        return {adv_name: adv_acc,
                auroc_name: auroc,
                aupr_name: aupr,
                entropy_name: ood_entropy,
                auroc_ent_name: auroc_entropy,
                aupr_ent_name: aupr_entropy
                }
