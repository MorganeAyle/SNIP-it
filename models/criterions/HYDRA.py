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


class HYDRA(General):
    def __init__(self, *args, arguments, limit, steps, **kwargs):
        self.arguments = arguments
        super(HYDRA, self).__init__(*args, **kwargs)
        self.post_init()
        self.loss = nn.CrossEntropyLoss()
        self.ood_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.steps = [limit - (limit - 0.5) * (0.5 ** i) for i in range(steps)] + [limit]
        # self.steps = [self.arguments["pruning_limit"]]

    def post_init(self):
        # get model
        arguments = self.arguments

        self.prune_model = deepcopy(self.model)

        # load data
        self.train_loader, _ = find_right_model(
            DATASETS, arguments['data_set'],
            arguments=arguments
        )

        self.ood_loader, _ = find_right_model(
            DATASETS, arguments['ood_data_set'],
            arguments=arguments
        )

        self.foo()

        # get optimizer
        self.optimizer = find_right_model(
            OPTIMS, arguments['optimizer'],
            params=self.prune_model.parameters(),
            lr=arguments['learning_rate']
        )

    def foo(self):
        for param in self.prune_model.parameters():
            param.requires_grad = False

        for name, layer in self.prune_model.named_modules():

            if name + ".weight" in self.prune_model.mask:
                print(name)
                requires_grad = True
                fan_in = calculate_fan_in(layer.weight)
                init_weights = torch.ones_like(layer.weight) * layer.weight * math.sqrt(6/fan_in) / torch.max(torch.abs(layer.weight))

                gov = nn.Parameter(init_weights, requires_grad=requires_grad)
                layer.gov = gov

                if isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(linear_forward, layer)

                elif isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(conv_forward, layer)

                print(layer)

    def prune(self, percentage, **kwargs):

        if len(self.steps) == 0:
            print("Pruning steps done")
            return

        print("Training mask adversarially...")
        percentage = self.steps.pop(0)
        print(percentage)

        for i in range(5):
            for batch in self.train_loader:
                self.optimizer.zero_grad()

                x, y = batch

                self.prune_model.eval()
                self.model.eval()

                adv_batch_size = int(self.arguments['batch_size'] / 2)
                x_to_adv = x[:adv_batch_size]
                y_to_adv = y[:adv_batch_size]
                adv_results, _ = construct_adversarial_examples(x_to_adv, y_to_adv, self.arguments['attack'], self.model,
                                                                self.device, self.arguments['epsilon'], exclude_wrong_predictions=False, targeted=False)
                _, advs, _ = adv_results
                x = torch.cat((advs.cpu(), x[adv_batch_size:]))

                self.prune_model.train()

                # unpack
                x, y = x.to(self.device).float(), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.prune_model(x)
                loss = self.loss.forward(out, y)

                loss.backward()
                self.optimizer.step()

        weight_masks = {}
        for name, layer in self.prune_model.named_modules():
            if name + ".weight" in self.prune_model.mask:
                weight_masks[name + ".weight"] = layer.gov.data
        all_scores = torch.cat([torch.flatten(x) for _, x in weight_masks.items()])

        num_params_to_keep = int(len(all_scores) * (1 - percentage))
        if num_params_to_keep < 1:
            num_params_to_keep += 1
        elif num_params_to_keep > len(all_scores):
            num_params_to_keep = len(all_scores)

        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for name, weight_mask in weight_masks.items():
            self.model.mask[name] = (weight_mask > acceptable_score).__and__(
                self.model.mask[name].bool()).float().to(self.device)

            # how much we wanna prune
            length_nonzero = float(self.model.mask[name].flatten().shape[0])

            cutoff = (self.model.mask[name] == 0).sum().item()

            print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)

        self.model.apply_weight_mask()
        print("Final prune percentage ", self.model.pruned_percentage)


    # def prune(self, percentage, **kwargs):
    #
    #     if len(self.steps) == 0:
    #         print("Pruning steps done")
    #         return
    #
    #     print("Training mask...")
    #
    #     percentage = self.steps.pop(0)
    #     print(percentage)
    #
    #     for batch, ood_batch in zip(self.train_loader, self.ood_loader):
    #         x, y = batch
    #
    #         # unpack
    #         x, y = x.to(self.device).float(), y.to(self.device)
    #
    #         self.optimizer.zero_grad()
    #
    #         # normal loss
    #         out = self.prune_model(x)
    #         loss = self.loss.forward(out, y)
    #
    #         # entropy normal
    #         in_loss = -self.ood_loss.forward(torch.log(out), torch.full_like(out, 0.1))
    #
    #         # ood loss
    #         ood_x, ood_y = ood_batch
    #         ood_x, ood_y = ood_x.to(self.device).float(), ood_y.to(self.device)
    #         out = self.prune_model(ood_x)
    #         ood_loss = self.ood_loss.forward(torch.log(out), torch.full_like(out, 0.1))
    #
    #         loss = 0.5 * loss + 0.25 * ood_loss + 0.25 * in_loss
    #
    #         loss.backward()
    #         self.optimizer.step()
    #
    #     weight_masks = {}
    #     for name, layer in self.prune_model.named_modules():
    #         if name + ".weight" in self.prune_model.mask:
    #             weight_masks[name + ".weight"] = layer.gov.data
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
