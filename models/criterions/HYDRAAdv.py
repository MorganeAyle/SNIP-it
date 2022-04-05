from models.criterions.HYDRA import HYDRA
from models import GeneralModel
from utils.model_utils import *
from utils.system_utils import *
import torch
from utils.attacks_utils import construct_adversarial_examples
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
import types
from utils.hydra_utils import linear_forward, conv_forward, calculate_fan_in
import math
from torch.distributions import Categorical
import numpy as np
from utils.metrics import calculate_auroc


class HYDRAAdv(HYDRA):
    def __init__(self, *args, arguments, limit, steps, **kwargs):
        self.arguments = arguments
        super(HYDRAAdv, self).__init__(arguments=arguments, limit=limit, steps=steps, *args, **kwargs)
        self.post_init()
        self.loss = nn.CrossEntropyLoss()
        self.ood_loss = torch.nn.KLDivLoss(reduction='batchmean')
        # self.steps = [limit - (limit - 0.5) * (0.5 ** i) for i in range(steps)] + [limit]
        self.steps = [self.arguments["pruning_limit"]]

    def prune(self, percentage, **kwargs):

        if len(self.steps) == 0:
            print("Pruning steps done")
            return

        self.foo()

        # get optimizer
        self.optimizer = find_right_model(
            OPTIMS, self.arguments['optimizer'],
            params=self.prune_model.parameters(),
            lr=self.arguments['learning_rate']
        )

        print("Training mask adversarially...")

        percentage = self.steps.pop(0)
        print(percentage)

        self.prune_model.eval()

        acc = []
        for batch in self.test_loader:
            self.handle_global_pruning(percentage)  #################################
            x, y = batch
            x, y = x.to(self.device).float(), y.to(self.device)
            out = self.prune_model(x).squeeze()
            predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
            correct = y.eq(predictions).sum().item()
            acc.append(correct / out.shape[0])
        print(np.mean(acc))

        self.prune_model.train()
        best_model = None
        best_acc = 0

        best_count = 0
        early_stopping = 5

        for i in range(self.arguments["epochs"]):
            for batch in self.train_loader_unnormalized:
                self.handle_global_pruning(percentage)  #################################

                self.prune_model.eval()

                x, y = batch

                adv_batch_size = int(self.arguments['batch_size'])
                x_to_adv = x[:adv_batch_size]
                y_to_adv = y[:adv_batch_size]
                adv_results, _ = construct_adversarial_examples(x_to_adv, y_to_adv, self.arguments['attack'], self.prune_model,
                                                                self.device, self.arguments['epsilon'], exclude_wrong_predictions=False, targeted=False)
                _, x_adv, _ = adv_results

                # normalize attacks
                new_x = []
                for image in x_adv:
                    if image.dim() > 3:
                        image = self.transform(image.squeeze(0)).unsqueeze(0)
                    else:
                        image = self.transform(image).unsqueeze(0)
                    new_x.append(image)

                # normalize clean input
                new_x = []
                for image in x:
                    if image.dim() > 3:
                        image = self.transform(image.squeeze(0)).unsqueeze(0)
                    else:
                        image = self.transform(image).unsqueeze(0)
                    new_x.append(image)
                x = torch.cat(new_x)

                self.prune_model.train()

                x, y = x.to(self.device).float(), y.to(self.device)
                x_adv = x_adv.to(self.device)

                self.optimizer.zero_grad()
                out = self.prune_model(x)
                loss = self.loss.forward(out, y)

                loss_robust = self.ood_loss(
                    F.log_softmax(self.prune_model(x_adv), dim=1), F.softmax(self.prune_model(x), dim=1)
                )

                loss += 6.0 * loss_robust

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.prune_model.eval()
            acc = []
            adv_acc = []
            for batch in self.test_loader:
                self.handle_global_pruning(percentage)  #################################
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                acc.append(correct / out.shape[0])

            # Validate on attacks
            for batch in self.test_loader_unnormalized:
                self.handle_global_pruning(percentage) ##############################
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)

                adv_results, _ = construct_adversarial_examples(x, y, self.arguments['attack'],
                                                                self.prune_model,
                                                                self.device, self.arguments['epsilon'],
                                                                exclude_wrong_predictions=False, targeted=False)
                _, x_adv, _ = adv_results

                new_x = []
                for image in x_adv:
                    if image.dim() > 3:
                        image = self.transform(image.squeeze(0)).unsqueeze(0)
                    else:
                        image = self.transform(image).unsqueeze(0)
                    new_x.append(image)
                x_adv = torch.cat(new_x)

                x_adv = x_adv.to(self.device)

                out = self.prune_model(x_adv).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                adv_acc.append(correct / out.shape[0])

            print('Epoch', i)
            print('Accuracy', np.mean(acc))
            print('Adversarial Accuracy', np.mean(adv_acc))

            if np.mean(adv_acc) > best_acc:
                print('Best model')
                best_model = deepcopy(self.prune_model)
                best_acc = np.mean(adv_acc)

                best_count = 0
                continue
            best_count += 1
            if best_count == early_stopping:
                break

            self.prune_model.train()

        self.prune_model = best_model

        self.handle_pruning(percentage)
