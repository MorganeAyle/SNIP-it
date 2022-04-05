from models.criterions.General import General
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


class Edgepop(General):
    def __init__(self, *args, arguments, limit, steps, **kwargs):
        self.arguments = arguments
        super(Edgepop, self).__init__(*args, **kwargs)
        self.post_init()
        self.loss = nn.CrossEntropyLoss()
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

    def foo(self):
        self.prune_model = deepcopy(self.model)
        for param in self.prune_model.parameters():
            if not isinstance(param, nn.BatchNorm2d):
                param.requires_grad = False

        for name, layer in self.prune_model.named_modules():

            if name + ".weight" in self.prune_model.mask:
                print(name)

                gov = nn.Parameter(torch.ones_like(layer.weight), requires_grad=True)
                torch.nn.init.kaiming_normal_(
                    gov.data, mode='fan_in', nonlinearity='relu'
                )
                layer.gov = gov
                layer.prune_rate = self.steps[0]

                if isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(linear_forward, layer)

                elif isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(conv_forward, layer)

                print(layer)

    def handle_pruning(self, percentage):
        weight_masks = {}
        for name, layer in self.prune_model.named_modules():
            if name + ".weight" in self.prune_model.mask:
                weight_masks[name + ".weight"] = layer.gov.data
        all_scores = torch.cat([torch.flatten(x) for _, x in weight_masks.items()])

        local = False

        if not local:
            num_params_to_keep = int(len(all_scores) * (1 - percentage))
            if num_params_to_keep < 1:
                num_params_to_keep += 1
            elif num_params_to_keep > len(all_scores):
                num_params_to_keep = len(all_scores)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

        for name, weight_mask in weight_masks.items():
            if local:
                # don't prune more or less than possible
                num_params_to_keep = int(len(torch.flatten(weight_mask)) * (1 - percentage))
                if num_params_to_keep < 1:
                    num_params_to_keep += 1
                elif num_params_to_keep > len(torch.flatten(weight_mask)):
                    num_params_to_keep = len(torch.flatten(weight_mask))

                # threshold
                threshold, _ = torch.topk(torch.flatten(weight_mask), num_params_to_keep, sorted=True)
                acceptable_score = threshold[-1]

            self.model.mask[name] = (weight_mask > acceptable_score).__and__(
                self.model.mask[name].bool()).float().to(self.device)

            # how much we wanna prune
            length_nonzero = float(self.model.mask[name].flatten().shape[0])

            cutoff = (self.model.mask[name] == 0).sum().item()

            print("pruning", name, cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)

        self.model.apply_weight_mask()
        print("Final prune percentage ", self.model.pruned_percentage)

    def handle_global_pruning(self, percentage):
        # for global pruning
        weight_masks = {}
        for name, layer in self.prune_model.named_modules():
            if name + ".weight" in self.prune_model.mask:
                weight_masks[name + ".weight"] = layer.gov.data
        all_scores = torch.cat([torch.flatten(x) for _, x in weight_masks.items()])
        num_params_to_keep = int(len(all_scores) * (1 - percentage))
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        for name, layer in self.prune_model.named_modules():
            if name + ".weight" in self.prune_model.mask:
                layer.prune_rate = (weight_masks[name + ".weight"] < acceptable_score).float().sum() / torch.numel(
                    layer.weight)

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

        print("Training mask...")

        percentage = self.steps.pop(0)
        print(percentage)

        self.prune_model.eval()

        acc = []
        for batch in self.test_loader:
            self.handle_global_pruning(percentage)
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

        for i in range(self.arguments["epochs"]):
            self.prune_model.train()
            for batch in self.train_loader:
                self.prune_model.eval()
                self.handle_global_pruning(percentage)
                self.prune_model.train()

                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.prune_model(x)
                loss = self.loss.forward(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.prune_model.eval()
            acc = []
            for batch in self.test_loader:
                self.handle_global_pruning(percentage)
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device)
                out = self.prune_model(x).squeeze()
                predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
                correct = y.eq(predictions).sum().item()
                acc.append(correct / out.shape[0])

            print('Epoch', i)
            print('Accuracy', np.mean(acc))

            if np.mean(acc) > best_acc:
                print('Best model')
                best_model = deepcopy(self.prune_model)
                best_acc = np.mean(acc)

        self.prune_model = best_model

        self.handle_pruning(percentage)