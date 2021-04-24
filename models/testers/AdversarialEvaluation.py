import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.attacks_utils import construct_adversarial_examples


class AdversarialEvaluation:

    """
    Performs evaluation to adversarial attacks
    """

    def __init__(self, train_loader, test_loader, model, loss, optimizer, device, arguments, **kwargs):
        self.args = arguments
        self.fb = fb
        self.device = device
        self.optimizer = optimizer
        self.loss = loss
        self.model = model
        self.loader_test = test_loader
        self.loader = train_loader

    def evaluate(self, plot=False, targeted=False, exclude_wrong_predictions=False, epsilons=[0.25, 0.5, 0.75, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10], curr_model=None):

        method = self.args['attack']

        if curr_model is None:
            model = self.model.eval().to(self.device)
        else:
            model = curr_model

        im, crit = next(iter(self.loader_test))
        adv_results, predictions = construct_adversarial_examples(im, crit, method, model, self.device, exclude_wrong_predictions, targeted, epsilons)
        _, advs, success = adv_results

        plt.title(method)

        sucess_rates = []
        for eps, eps_adv, eps_success in zip(epsilons, advs, success):
            attack_success = (eps_success.float().sum() / len(eps_success)).item()
            adv_equality = (model.forward(eps_adv).argmax(dim=-1) == predictions)
            predicted_same_as_model = (adv_equality.float().sum() / len(eps_success)).item()

            sucess_rates.append(attack_success)

            if curr_model is None:
                print("EPSILON", eps,
                    "Successes attack", attack_success,
                    "same prediction as model", predicted_same_as_model,
                    "bounds adver", eps_adv.min().item(), eps_adv.max().item(),
                    "norm", np.mean([torch.norm(eps_adv_ - im_, p=2).item() for eps_adv_, im_ in zip(eps_adv, im)]))

            if plot:
                self._plot(adv_equality, eps, eps_adv, im, model)

        return epsilons, np.array(sucess_rates)
