import copy
import os
import types
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.criterions.General import General
from models.networks.assisting_layers.GateDecoratorLayers import GatedBatchNorm
from utils.constants import SNIP_BATCH_ITERATIONS, RESULTS_DIR, OUTPUT_DIR
from utils.data_utils import lookahead_type, lookahead_finished
from utils.snip_utils import group_snip_forward_linear, group_snip_conv2d_forward
from utils.attacks_utils import construct_adversarial_examples


class StructuredSNR(General):

    """
    Original creation from our paper:  https://arxiv.org/abs/2006.00896
    Implements SNAP (structured), which is one of the steps from the algorithm SNAP-it
    Additionally, this class contains most of the code the actually reduce pytorch tensors, in order to obtain speedup
    """

    def __init__(self, *args, **kwargs):
        super(StructuredSNR, self).__init__(*args, **kwargs)

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, *args, **kwargs):
        raise NotImplementedError

    def prune(self, percentage, train_loader=None, manager=None, **kwargs):

        all_scores, grads_abs, log10, norm_factor, vec_shapes = self.get_weight_saliencies(train_loader)

        self.handle_pruning(all_scores, grads_abs, norm_factor, percentage, vec_shapes)

    def handle_pruning(self, all_scores, grads_abs, norm_factor, percentage, vec_shapes=None):

        summed_weights = sum([np.prod(x.shape) for name, x in self.model.named_parameters() if "W_mu" in name])
        num_nodes_to_keep = int(len(all_scores) * (1 - percentage))

        # dont prune more or less than is available
        if num_nodes_to_keep > len(all_scores):
            num_nodes_to_keep = len(all_scores)
        elif num_nodes_to_keep == 0:
            num_nodes_to_keep = 1

        # threshold
        threshold, _ = torch.topk(all_scores, num_nodes_to_keep, sorted=True)
        del _
        acceptable_score = threshold[-1]

        # prune
        summed_pruned = 0
        toggle_row_column = False  # False means output
        cutoff = 0
        length_nonzero = 0
        grads_abs.keys()
        for ((_, name), grad), (first, last) in lookahead_finished(grads_abs.items()):

            binary_keep_neuron_vector = ((grad / norm_factor) >= acceptable_score).float().to(self.device)
            corresponding_weight_parameter = [val for key, val in self.model.named_parameters() if key == (name + '.W_mu')][0]
            corresponding_weight_parameter2 = [val for key, val in self.model.named_parameters() if key == (name + '.W_rho')][0]
            is_conv = len(corresponding_weight_parameter.shape) > 2
            corresponding_module: nn.Module = \
                [val for key, val in self.model.named_modules() if key == name][0]

            # ensure not disconnecting
            if binary_keep_neuron_vector.sum() == 0:
                best_index = torch.argmax(grad)
                binary_keep_neuron_vector[best_index] = 1

            if first or last:
                # noinspection PyTypeChecker
                length_nonzero = self.handle_outer_layers(binary_keep_neuron_vector,
                                                          first,
                                                          is_conv,
                                                          last,
                                                          length_nonzero,
                                                          corresponding_module,
                                                          name,
                                                          corresponding_weight_parameter,
                                                          corresponding_weight_parameter2)

                if first and not toggle_row_column:
                    print(name, "pruning percentage", 1 - binary_keep_neuron_vector.sum() / len(binary_keep_neuron_vector))
                if last and toggle_row_column:
                    print(name, "pruning percentage",
                          1 - binary_keep_neuron_vector.sum() / len(binary_keep_neuron_vector))
            else:

                cutoff, length_nonzero = self.handle_middle_layers(binary_keep_neuron_vector,
                                                                   cutoff,
                                                                   is_conv,
                                                                   length_nonzero,
                                                                   corresponding_module,
                                                                   name,
                                                                   toggle_row_column,
                                                                   corresponding_weight_parameter,
                                                                   corresponding_weight_parameter2)

                if not toggle_row_column:
                    print(name, "pruning percentage", 1 - binary_keep_neuron_vector.sum() / len(binary_keep_neuron_vector))

            toggle_row_column = not toggle_row_column

        last = None
        for name, module in self.model.named_modules():
            if last is None and 'conv' in name:
                last = module
            else:
                if 'flatten' in name:
                    module.update_input_dim(int(last.out_channels * 3 * 3))
                if 'conv' in name:
                    last = module

        for line in str(self.model).split("\n"):
            if "BatchNorm" in line or "Conv" in line or "Linear" in line or "AdaptiveAvg" in line or "Sequential" in line:
                print(line)
        print("final percentage after snap:", 1 - sum([np.prod(x.shape) for name, x in self.model.named_parameters() if "W_mu" in name]) / summed_weights)

        # self.model.apply_weight_mask()
        self.cut_lonely_connections()

        for param in self.model.parameters():
            param.requires_grad = True

    def handle_middle_layers(self,
                             binary_vector,
                             cutoff,
                             is_conv,
                             length_nonzero,
                             module,
                             name,
                             toggle_row_column,
                             weight, weight2):

        indices = binary_vector.bool()
        length_nonzero_before = int(np.prod(weight.shape))
        n_remaining = binary_vector.sum().item()
        if not toggle_row_column:
            self.handle_output(indices,
                               is_conv,
                               module,
                               n_remaining,
                               name,
                               weight, weight2)

        else:
            cutoff, length_nonzero = self.handle_input(cutoff,
                                                       indices,
                                                       is_conv,
                                                       length_nonzero,
                                                       module,
                                                       n_remaining,
                                                       name,
                                                       weight, weight2)

        cutoff += (length_nonzero_before - int(np.prod(weight.shape)))
        return cutoff, length_nonzero

    def handle_input(self, cutoff, indices, is_conv, length_nonzero, module, n_remaining, name, weight, weight2):
        """ shrinks a input dimension """
        module.update_input_dim(n_remaining)
        length_nonzero = int(np.prod(weight.shape))
        cutoff = 0
        if is_conv:
            weight.data = weight[:, indices, :, :]
            weight2.data = weight2[:, indices, :, :]
            try:
                weight.grad.data = weight.grad.data[:, indices, :, :]
                weight2.grad.data = weight2.grad.data[:, indices, :, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][:, indices, :, :]
        else:
            if ((weight.shape[1] % indices.shape[0]) == 0) and not (weight.shape[1] == indices.shape[0]):
                ratio = weight.shape[1] // indices.shape[0]
                module.update_input_dim(n_remaining * ratio)
                new_indices = torch.repeat_interleave(indices, ratio)
                weight.data = weight[:, new_indices]
                weight2.data = weight2[:, new_indices]
                if name in self.model.mask:
                    self.model.mask[name] = self.model.mask[name][:, new_indices]
                try:
                    weight.grad.data = weight.grad.data[:, new_indices]
                    weight2.grad.data = weight2.grad.data[:, new_indices]
                except AttributeError:
                    pass
            else:
                weight.data = weight[:, indices]
                weight2.data = weight2[:, indices]
                try:
                    weight.grad.data = weight.grad.data[:, indices]
                    weight2.grad.data = weight2.grad.data[:, indices]
                except AttributeError:
                    pass
                if name in self.model.mask:
                    self.model.mask[name] = self.model.mask[name][:, indices]
        if self.model.is_tracking_weights:
            raise NotImplementedError
        return cutoff, length_nonzero

    def handle_output(self, indices, is_conv, module, n_remaining, name, weight, weight2):
        """ shrinks a output dimension """

        module.update_output_dim(n_remaining)
        # self.handle_batch_norm(indices, n_remaining, name)
        if is_conv:
            weight.data = weight[indices, :, :, :]
            weight2.data = weight2[indices, :, :, :]
            try:
                weight.grad.data = weight.grad.data[indices, :, :, :]
                weight2.grad.data = weight2.grad.data[indices, :, :, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][indices, :, :, :]
        else:
            weight.data = weight[indices, :]
            weight2.data = weight2[indices, :]
            try:
                weight.grad.data = weight.grad.data[indices, :]
                weight2.grad.data = weight2.grad.data[indices, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][indices, :]
        self.handle_bias(indices, name)
        if self.model.is_tracking_weights:
            raise NotImplementedError

    def handle_bias(self, indices, name):
        """ shrinks a bias """
        bias = [val for key, val in self.model.named_parameters() if key == name + '.bias_mu'][0]
        bias.data = bias[indices]
        bias = [val for key, val in self.model.named_parameters() if key == name + '.bias_rho'][0]
        bias.data = bias[indices]
        try:
            bias.grad.data = bias.grad.data[indices]
        except AttributeError:
            pass

    def handle_batch_norm(self, indices, n_remaining, name):
        """ shrinks a batchnorm layer """

        batchnorm = [val for key, val in self.model.named_modules() if
                     key == name.split(".weight")[0][:-1] + str(int(name.split(".weight")[0][-1]) + 1)][0]
        if isinstance(batchnorm, (nn.BatchNorm2d, nn.BatchNorm1d, GatedBatchNorm)):
            batchnorm.num_features = n_remaining
            from_size = len(batchnorm.bias.data)
            batchnorm.bias.data = batchnorm.bias[indices]
            batchnorm.weight.data = batchnorm.weight[indices]
            try:
                batchnorm.bias.grad.data = batchnorm.bias.grad[indices]
                batchnorm.weight.grad.data = batchnorm.weight.grad[indices]
            except TypeError:
                pass
            if hasattr(batchnorm, "gate"):
                batchnorm.gate.data = batchnorm.gate.data[indices]
                batchnorm.gate.grad.data = batchnorm.gate.grad.data[indices]
                batchnorm.bn.num_features = n_remaining
            for buffer in batchnorm.buffers():
                if buffer.data.shape == indices.shape:
                    buffer.data = buffer.data[indices]
            print(f"trimming nodes in layer {name} from {from_size} to {len(batchnorm.bias.data)}")

    def handle_outer_layers(self,
                            binary_vector,
                            first,
                            is_conv,
                            last,
                            length_nonzero,
                            module,
                            name,
                            param, param2):

        n_remaining = binary_vector.sum().item()
        if first:
            length_nonzero = int(np.prod(param.shape))
            self.handle_output(binary_vector.bool(), is_conv, module, n_remaining, name, param, param2)
        if last:
            length_nonzero = int(np.prod(param.shape))
            self.handle_input(None, binary_vector.bool(), is_conv, length_nonzero, module, n_remaining, name, param, param2)
        return length_nonzero

    def print_layer_progress(self, cutoff, grads_abs, length_nonzero, name, summed_pruned, toggle, weight):
        if not toggle:
            if len(grads_abs) == 2:
                cutoff /= 2
            summed_pruned += cutoff
            print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
        return cutoff, summed_pruned

    def get_weight_saliencies(self, train_loader):

        # copy network
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.model.zero_grad()

        for param in self.model.parameters():
            param.requires_grad = False

        last = None

        # gather elasticities
        grads_abs = OrderedDict()
        for name, layer in self.model.named_modules():
            if 'conv' in name:
                if last is None:
                    grads_abs[(1, name)] = (torch.abs(
                        layer.W_mu.data) / torch.log1p(torch.exp(layer.W_rho.data))).sum((1, 2, 3)) / torch.numel(layer.W_mu.data)
                    last = (torch.abs(
                        layer.W_mu.data) / torch.log1p(torch.exp(layer.W_rho.data))).sum((1, 2, 3)) / torch.numel(layer.W_mu.data)
                else:
                    grads_abs[(0, name)] = last
                    grads_abs[(1, name)] = (torch.abs(
                        layer.W_mu.data) / torch.log1p(torch.exp(layer.W_rho.data))).sum((1, 2, 3)) / torch.numel(layer.W_mu.data)
                    last = grads_abs[(1, name)] = (torch.abs(
                        layer.W_mu.data) / torch.log1p(torch.exp(layer.W_rho.data))).sum((1, 2, 3)) / torch.numel(layer.W_mu.data)

            elif 'fc' in name:
                grads_abs[(0, name)] = last
                grads_abs[(1, name)] = (torch.abs(
                    layer.W_mu.data) / torch.log1p(torch.exp(layer.W_rho.data))).sum(-1) / torch.numel(layer.W_mu.data)
                last = (torch.abs(
                    layer.W_mu.data) / torch.log1p(torch.exp(layer.W_rho.data))).sum(-1) / torch.numel(layer.W_mu.data)

        # del grads_abs[list(grads_abs.keys())[0]]
        del grads_abs[list(grads_abs.keys())[-1]]

        self.model = self.model.to(self.device)
        self.model = self.model.train()

        all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])

        log10 = all_scores.sort().values.log10()
        return all_scores, grads_abs, log10, 1.0, [x.shape[0] for x in grads_abs.values()]

    def insert_governing_variables(self, net):
        """ inserts c vectors in all parameters """

        govs = []
        gov_in = None
        gov_out = None
        do_avg_pool = 0
        for layer, _ in lookahead_type(net.modules()):

            is_conv = isinstance(layer, nn.Conv2d)
            is_fc = isinstance(layer, nn.Linear)
            is_avgpool = isinstance(layer, nn.AdaptiveAvgPool2d)

            if is_avgpool:
                do_avg_pool = int(np.prod(layer.output_size))

            elif is_conv or is_fc:

                out_dim, in_dim = layer.weight.shape[:2]

                if gov_in is None:

                    gov_in = nn.Parameter(torch.ones(in_dim).to(self.device), requires_grad=True)
                    govs.append(gov_in)

                else:
                    gov_in = gov_out

                gov_out = nn.Parameter(torch.ones(out_dim).to(self.device), requires_grad=True)
                govs.append(gov_out)

                # insert variables
                layer.gov_out = gov_out
                layer.gov_in = gov_in

                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

            # substitute activation function
            if is_fc:
                if do_avg_pool > 0:
                    layer.do_avg_pool = do_avg_pool
                    do_avg_pool = 0
                layer.forward = types.MethodType(group_snip_forward_linear, layer)
            if is_conv:
                layer.forward = types.MethodType(group_snip_conv2d_forward, layer)

        return govs
