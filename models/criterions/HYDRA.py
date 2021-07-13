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


class HYDRA(General):
    def __init__(self, *args, arguments, limit, steps, **kwargs):
        self.arguments = arguments
        super(HYDRA, self).__init__(*args, **kwargs)
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

        # self.foo()
        #
        # # get optimizer
        # self.optimizer = find_right_model(
        #     OPTIMS, arguments['optimizer'],
        #     params=self.prune_model.parameters(),
        #     lr=arguments['learning_rate']
        # )

    def foo(self):
        self.prune_model = deepcopy(self.model)
        for param in self.prune_model.parameters():
            param.requires_grad = False

        for name, layer in self.prune_model.named_modules():

            if name + ".weight" in self.prune_model.mask:
                print(name)
                requires_grad = True
                fan_in = calculate_fan_in(layer.weight)
                init_weights = layer.weight * math.sqrt(6/fan_in) / torch.max(torch.abs(layer.weight))

                # init_weights = torch.ones_like(layer.weight)

                gov = nn.Parameter(init_weights, requires_grad=requires_grad)
                layer.gov = gov
                # torch.nn.init.xavier_uniform_(layer.gov.data)

                torch.nn.init.kaiming_normal_(
                    layer.gov.data, mode='fan_in', nonlinearity='relu'
                )

                if isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(linear_forward, layer)

                elif isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(conv_forward, layer)

                print(layer)

    # def prune(self, percentage, **kwargs):
    #
    #     if len(self.steps) == 0:
    #         print("Pruning steps done")
    #         return
    #
    #     self.foo()
    #
    #     # get optimizer
    #     self.optimizer = find_right_model(
    #         OPTIMS, self.arguments['optimizer'],
    #         params=self.prune_model.parameters(),
    #         lr=self.arguments['learning_rate']
    #     )
    #
    #     print("Training mask ...")
    #     percentage = self.steps.pop(0)
    #     print(percentage)
    #
    #     self.prune_model.train()
    #
    #     for i in range(100):
    #         for batch in self.train_loader:
    #
    #             x, y = batch
    #
    #             # unpack
    #             x, y = x.to(self.device).float(), y.to(self.device)
    #
    #             self.optimizer.zero_grad()
    #             out = self.prune_model(x)
    #             loss = self.loss.forward(out, y)
    #
    #             loss.backward()
    #             self.optimizer.step()
    #
    #         self.prune_model.eval()
    #         acc = []
    #         ood_true = np.zeros(0)
    #         ood_preds = np.zeros(0)
    #         for batch in self.test_loader:
    #             x, y = batch
    #             x, y = x.to(self.device).float(), y.to(self.device)
    #             out = self.prune_model(x).squeeze()
    #             predictions = out.argmax(dim=-1, keepdim=True).view_as(y)
    #             correct = y.eq(predictions).sum().item()
    #             acc.append(correct / out.shape[0])
    #
    #             probs = F.softmax(out, dim=-1)
    #             preds, _ = torch.max(probs, dim=-1)
    #             ood_true = np.concatenate((ood_true, np.ones(len(preds))))
    #             ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))
    #
    #         for batch in self.ood_loader_test:
    #             x, y = batch
    #             x, y = x.to(self.device).float(), y.to(self.device)
    #             out = self.prune_model(x).squeeze()
    #
    #             probs = F.softmax(out, dim=-1)
    #             preds, _ = torch.max(probs, dim=-1)
    #             ood_true = np.concatenate((ood_true, np.zeros(len(preds))))
    #             ood_preds = np.concatenate((ood_preds, preds.reshape((-1))))
    #
    #         print('Epoch', i)
    #         print('Accuracy', np.mean(acc))
    #         print('AUROC', calculate_auroc(ood_true, ood_preds))
    #
    #         self.prune_model.train()
    #
    #     weight_masks = {}
    #     for name, layer in self.prune_model.named_modules():
    #         if name + ".weight" in self.prune_model.mask:
    #             weight_masks[name + ".weight"] = torch.abs(layer.gov.data)
    #     all_scores = torch.cat([torch.flatten(x) for _, x in weight_masks.items()])
    #
    #     num_params_to_keep = int(len(all_scores) * (1 - percentage))
    #     if num_params_to_keep < 1:
    #         num_params_to_keep += 1
    #     elif num_params_to_keep > len(all_scores):
    #         num_params_to_keep = len(all_scores)
    #
    #     threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    #     acceptable_score = threshold[-1]
    #
    #     for name, weight_mask in weight_masks.items():
    #         self.model.mask[name] = (weight_mask > acceptable_score).__and__(
    #             self.model.mask[name].bool()).float().to(self.device)
    #
    #         # how much we wanna prune
    #         length_nonzero = float(self.model.mask[name].flatten().shape[0])
    #
    #         cutoff = (self.model.mask[name] == 0).sum().item()
    #
    #         print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
    #
    #     self.model.apply_weight_mask()
    #     print("Final prune percentage ", self.model.pruned_percentage)

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

        for i in range(80):
            for batch, ood_batch in zip(self.train_loader, self.ood_loader_train):
                if len(batch[0]) == len(ood_batch[0]):

                    # normal loss
                    x, y = batch
                    x, y = x.to(self.device).float(), y.to(self.device)
                    out = self.prune_model(x)
                    loss = self.loss.forward(out, y)

                    # ood loss
                    # ood_x, ood_y = ood_batch
                    # ood_x, ood_y = ood_x.to(self.device).float(), ood_y.to(self.device)
                    # ood_out = self.prune_model(ood_x)
                    # ood_loss = self.loss.forward(ood_out, torch.ones_like(ood_y)/ood_y.shape[-1])
                    #
                    # loss += 0.5 * ood_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            self.prune_model.eval()
            acc = []
            ood_true = np.zeros(0)
            ood_preds = np.zeros(0)
            for batch in self.test_loader:
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

            # if np.mean(acc) > best_acc:
            #     print('Best model')
            #     best_model = deepcopy(self.prune_model)
            #     best_acc = np.mean(acc)

            self.prune_model.train()

        self.prune_model = best_model

        weight_masks = {}
        for name, layer in self.prune_model.named_modules():
            if name + ".weight" in self.prune_model.mask:
                weight_masks[name + ".weight"] = torch.abs(layer.gov.data)
        all_scores = torch.cat([torch.flatten(x) for _, x in weight_masks.items()])

        num_params_to_keep = int(len(all_scores) * (1 - percentage))
        if num_params_to_keep < 1:
            num_params_to_keep += 1
        elif num_params_to_keep > len(all_scores):
            num_params_to_keep = len(all_scores)

        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        print(acceptable_score)
        print(all_scores.max())
        print(all_scores.min())

        for name, weight_mask in weight_masks.items():
            self.model.mask[name] = (weight_mask > acceptable_score).__and__(
                self.model.mask[name].bool()).float().to(self.device)

            # how much we wanna prune
            length_nonzero = float(self.model.mask[name].flatten().shape[0])

            cutoff = (self.model.mask[name] == 0).sum().item()

            print("pruning", name, cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)

        self.model.apply_weight_mask()
        print("Final prune percentage ", self.model.pruned_percentage)

    # def prune(self, percentage, **kwargs):
    #
    #     if len(self.steps) == 0:
    #         print("Pruning steps done")
    #         return
    #
    #     self.foo()
    #
    #     # get optimizer
    #     self.optimizer = find_right_model(
    #         OPTIMS, self.arguments['optimizer'],
    #         params=self.prune_model.parameters(),
    #         lr=self.arguments['learning_rate']
    #     )
    #
    #     print("Training mask adversarially...")
    #     percentage = self.steps.pop(0)
    #     print(percentage)
    #
    #     self.prune_model.train()
    #     # self.model.eval()
    #
    #     for i in range(100):
    #         for batch in self.train_loader:
    #             self.optimizer.zero_grad()
    #
    #             self.prune_model.eval()
    #
    #             x, y = batch
    #
    #             adv_batch_size = int(self.arguments['batch_size'])
    #             x_to_adv = x[:adv_batch_size]
    #             y_to_adv = y[:adv_batch_size]
    #             adv_results, _ = construct_adversarial_examples(x_to_adv, y_to_adv, self.arguments['attack'], self.prune_model,
    #                                                             self.device, self.arguments['epsilon'], exclude_wrong_predictions=False, targeted=False)
    #             _, x_adv, _ = adv_results
    #             x_adv = x_adv.to(self.device)
    #
    #             self.prune_model.train()
    #
    #             # unpack
    #             x, y = x.to(self.device).float(), y.to(self.device)
    #
    #             self.optimizer.zero_grad()
    #             out = self.prune_model(x)
    #             loss = self.loss.forward(out, y)
    #
    #             loss_robust = self.ood_loss(
    #                 F.log_softmax(self.prune_model(x_adv), dim=1), F.softmax(self.prune_model(x), dim=1)
    #             )
    #
    #             loss += 6.0 * loss_robust
    #
    #             loss.backward()
    #             self.optimizer.step()
    #
    #     weight_masks = {}
    #     for name, layer in self.prune_model.named_modules():
    #         if name + ".weight" in self.prune_model.mask:
    #             weight_masks[name + ".weight"] = torch.abs(layer.gov.data)
    #     all_scores = torch.cat([torch.flatten(x) for _, x in weight_masks.items()])
    #
    #     num_params_to_keep = int(len(all_scores) * (1 - percentage))
    #     if num_params_to_keep < 1:
    #         num_params_to_keep += 1
    #     elif num_params_to_keep > len(all_scores):
    #         num_params_to_keep = len(all_scores)
    #
    #     threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    #     acceptable_score = threshold[-1]
    #
    #     for name, weight_mask in weight_masks.items():
    #         self.model.mask[name] = (weight_mask > acceptable_score).__and__(
    #             self.model.mask[name].bool()).float().to(self.device)
    #
    #         # how much we wanna prune
    #         length_nonzero = float(self.model.mask[name].flatten().shape[0])
    #
    #         cutoff = (self.model.mask[name] == 0).sum().item()
    #
    #         print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
    #
    #     self.model.apply_weight_mask()
    #     print("Final prune percentage ", self.model.pruned_percentage)

    # def prune(self, percentage, **kwargs):
    #
    #     if len(self.steps) == 0:
    #         print("Pruning steps done")
    #         return
    #
    #     self.foo()
    #
    #     # get optimizer
    #     self.optimizer = find_right_model(
    #         OPTIMS, self.arguments['optimizer'],
    #         params=self.prune_model.parameters(),
    #         lr=self.arguments['learning_rate']
    #     )
    #
    #     print("Training mask...")
    #
    #     percentage = self.steps.pop(0)
    #     print(percentage)
    #
    #     self.prune_model.train()
    #     self.prune_model.add_hooks()
    #
    #     for _ in range(2):
    #         for batch, ood_batch in zip(self.train_loader, self.ood_loader):
    #             if len(batch[0]) == len(ood_batch[0]):
    #                 x, y = batch
    #
    #                 # unpack
    #                 x, y = x.to(self.device).float(), y.to(self.device)
    #
    #                 self.optimizer.zero_grad()
    #
    #                 # normal loss
    #                 out = self.prune_model(x)
    #
    #                 loss = self.loss.forward(out, y)
    #                 loss += Categorical(F.softmax(out, dim=-1)).entropy().squeeze().mean()
    #                 # loss = - self.ood_loss(
    #                 #     F.log_softmax(out, dim=1), torch.full_like(out, 0.1)
    #                 # )
    #
    #                 # ood loss
    #                 ood_x, ood_y = ood_batch
    #                 ood_x, ood_y = ood_x.to(self.device).float(), ood_y.to(self.device)
    #                 # ood_loss = self.ood_loss(
    #                 #     F.log_softmax(self.prune_model(ood_x), dim=1), torch.full_like(out, 0.1)
    #                 # )
    #                 # ood_loss = - self.ood_loss(
    #                 #     F.log_softmax(self.prune_model(ood_x), dim=1), F.softmax(self.prune_model(x), dim=1)
    #                 # )
    #                 ood_loss = - Categorical(F.softmax(self.prune_model(ood_x), dim=-1)).entropy().squeeze().mean()
    #
    #                 loss += ood_loss
    #                 # loss = 10 * ood_loss
    #
    #                 loss.backward()
    #                 self.optimizer.step()
    #
    #
    #     weight_masks = {}
    #     for name, layer in self.prune_model.named_modules():
    #         if name + ".weight" in self.prune_model.mask:
    #             weight_masks[name + ".weight"] = torch.abs(layer.gov.data)
    #     all_scores = torch.cat([torch.flatten(x) for _, x in weight_masks.items()])
    #
    #     num_params_to_keep = int(len(all_scores) * (1 - percentage))
    #     if num_params_to_keep < 1:
    #         num_params_to_keep += 1
    #     elif num_params_to_keep > len(all_scores):
    #         num_params_to_keep = len(all_scores)
    #
    #     threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    #     acceptable_score = threshold[-1]
    #
    #     for name, weight_mask in weight_masks.items():
    #         self.model.mask[name] = (weight_mask > acceptable_score).__and__(
    #             self.model.mask[name].bool()).float().to(self.device)
    #
    #         # how much we wanna prune
    #         length_nonzero = float(self.model.mask[name].flatten().shape[0])
    #
    #         cutoff = (self.model.mask[name] == 0).sum().item()
    #
    #         print("pruning", name, cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
    #
    #     self.model.apply_weight_mask()
    #     print("Final prune percentage ", self.model.pruned_percentage)
