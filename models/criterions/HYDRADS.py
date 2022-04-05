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


class HYDRADS(HYDRA):
    def __init__(self, *args, arguments, limit, steps, **kwargs):
        self.arguments = arguments
        super(HYDRADS, self).__init__(arguments=arguments, limit=limit, steps=steps, *args, **kwargs)
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

        print("Training mask DS...")

        percentage = self.steps.pop(0)
        print(percentage)

        self.prune_model.eval()

        acc = []
        for batch in self.test_loader:
            self.handle_global_pruning(percentage)  ###############################
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

        for i in range(self.arguments["epochs"]):
            for batch in self.train_loader:
                self.handle_global_pruning(percentage)  ###############################

                self.prune_model.train()

                x, y = batch
                x += (torch.rand(x.shape) > 0.5).float() * torch.normal(torch.zeros(x.shape), torch.ones(x.shape))

                x, y = x.to(self.device).float(), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.prune_model(x)
                loss = self.loss.forward(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.prune_model.eval()
            acc = []
            ds_acc = []
            for batch in self.test_loader:
                self.handle_global_pruning(percentage)  ###############################
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                acc.append(correct / out.shape[0])

            for batch in self.test_loader:
                self.handle_global_pruning(percentage)  ###############################
                x, y = batch
                x += (torch.rand(x.shape) > 0.5).float() * torch.normal(torch.zeros(x.shape), torch.ones(x.shape))
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                ds_acc.append(correct / out.shape[0])

            print('Epoch', i)
            print('Accuracy', np.mean(acc))
            print('DS Accuracy', np.mean(ds_acc))

            if np.mean(ds_acc) > best_acc:
                print('Best model')
                best_model = deepcopy(self.prune_model)
                best_acc = np.mean(ds_acc)

        self.prune_model = best_model

        self.handle_pruning(percentage)
