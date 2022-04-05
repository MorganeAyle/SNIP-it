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


class HYDRAOOD(HYDRA):
    def __init__(self, *args, arguments, limit, steps, **kwargs):
        self.arguments = arguments
        super(HYDRAOOD, self).__init__(arguments=arguments, limit=limit, steps=steps, *args, **kwargs)
        self.post_init()
        self.loss = nn.CrossEntropyLoss()
        self.ood_loss = torch.nn.KLDivLoss(reduction='batchmean')
        # self.steps = [limit - (limit - 0.5) * (0.5 ** i) for i in range(steps)] + [limit]
        self.steps = [self.arguments["pruning_limit"]]

    def post_init(self):
        # get model
        arguments = self.arguments

        # load data
        self.train_loader, self.test_loader = find_right_model(
            DATASETS, arguments['data_set'],
            arguments=arguments
        )

        self.ood_loader_train, self.ood_loader_test = find_right_model(
            DATASETS, arguments['ood_data_set'],
            arguments=arguments
        )

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

        print("Training mask OOD...")

        percentage = self.steps.pop(0)
        print(percentage)

        self.prune_model.eval()

        acc = []
        for batch in self.test_loader:
            self.handle_global_pruning(percentage) #################################
            x, y = batch
            x, y = x.to(self.device).float(), y.to(self.device)
            out = self.prune_model(x).squeeze()
            predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
            correct = y.eq(predictions).sum().item()
            acc.append(correct / out.shape[0])
        print(np.mean(acc))

        self.prune_model.train()
        best_auroc = 0
        best_model = None
        best_acc = 0
        ood_loss = nn.KLDivLoss(reduction='batchmean')

        best_count = 0
        early_stopping = 5

        for i in range(self.arguments["epochs"]):
            for batch, ood_batch in zip(self.train_loader, self.ood_loader_train):
                if len(batch[0]) == len(ood_batch[0]):
                    self.handle_global_pruning(percentage) ###############################

                    if i < 0:
                        # normal loss
                        x, y = batch
                        x, y = x.to(self.device).float(), y.to(self.device)
                        out = self.prune_model(x)
                        # loss = F.nll_loss(out, y)
                        loss = self.loss.forward(out, y)

                    else:
                        data = torch.cat((batch[0], ood_batch[0]), 0)
                        target = batch[1]

                        data, target = data.to(self.device).float(), target.to(self.device)

                        # forward
                        out = self.prune_model(data)

                        loss = F.cross_entropy(out[:len(batch[0])], target)
                        # cross-entropy from softmax distribution to uniform distribution
                        loss += 0.5 * -(out[len(batch[0]):].mean(1) - torch.logsumexp(out[len(batch[0]):], dim=1)).mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            self.prune_model.eval()
            acc = []
            adv_acc = []
            ood_true = np.zeros(0)
            ood_preds = np.zeros(0)
            for batch in self.test_loader:
                self.handle_global_pruning(percentage) #################################
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                acc.append(correct / out.shape[0])

                probs = F.softmax(out, dim=-1)
                preds, _ = torch.max(probs, dim=-1)
                preds = preds.detach().cpu()
                ood_true = np.concatenate((ood_true, np.ones(len(preds))))
                ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))

            for batch in self.ood_loader_test:
                self.handle_global_pruning(percentage) ####################################
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()

                probs = F.softmax(out, dim=-1)
                preds, _ = torch.max(probs, dim=-1)
                preds = preds.detach().cpu()
                ood_true = np.concatenate((ood_true, np.zeros(len(preds))))
                ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))

            print('Epoch', i)
            print('Accuracy', np.mean(acc))
            print('AUROC', calculate_auroc(ood_true, ood_preds))

            if calculate_auroc(ood_true, ood_preds) > best_auroc:
                print('Best model')
                best_model = deepcopy(self.prune_model)
                best_auroc = calculate_auroc(ood_true, ood_preds)

                best_count = 0
                continue
            best_count += 1
            if best_count == early_stopping:
                break

            # if np.mean(acc) > best_acc:
            #     print('Best model')
            #     best_model = deepcopy(self.prune_model)
            #     best_acc = np.mean(acc)

            self.prune_model.train()

        self.prune_model = best_model

        self.handle_pruning(percentage)
