import copy
import types
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from models.networks.assisting_layers.GateDecoratorLayers import GatedBatchNorm

from glow.SNIP import SNIP
from utils.constants import SNIP_BATCH_ITERATIONS
from utils.data_utils import lookahead_type, lookahead_finished
import numpy as np
from utils.snip_utils import group_snip_forward_linear, group_snip_conv2d_forward


class John(SNIP):
    """
    Adapted implementation of GraSP from the paper:
    Picking Winning Tickets Before Training by Preserving Gradient Flow
    https://arxiv.org/abs/2002.07376
    from the authors' github:
    https://github.com/alecwangcq/GraSP
    """

    def __init__(self, generative=False, img_size=28, nbins=2**5, channels=3, loss_f=None, *args, **kwargs):
        super(John, self).__init__(*args, **kwargs)
        self.generative = generative
        self.img_size = img_size
        self.nbins = nbins
        self.channels = channels
        self.loss_f = loss_f

    def get_prune_indices(self, *args, **kwargs):
        raise NotImplementedError

    def get_grow_indices(self, percentage, all_scores, grads_abs):
        # don't prune more or less than possible
        pass

    def cut_lonely_connections(self):
        govs = []
        gov_in = None
        gov_out = None
        do_avg_pool = 0
        for layer, (is_conv, next_is_conv) in lookahead_type(self.model.modules()):
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
            # substitute activation function
            if is_fc:
                if do_avg_pool > 0:
                    layer.do_avg_pool = do_avg_pool
                    do_avg_pool = 0
                # layer.forward = types.MethodType(group_snip_forward_linear, layer)
            # if is_conv:
            #     layer.forward = types.MethodType(group_snip_conv2d_forward, layer)
        indices = {}
        idx = 0
        for id, layer in self.model.mask.items():
            if 'conv' in id:
                # input
                input = []
                for i in range(layer.shape[1]):
                    if len(torch.nonzero(layer[:, i, :, :])) == 0:
                        input.append(0)
                    else:
                        input.append(1)
                # output
                output = []
                for i in range(layer.shape[0]):
                    if len(torch.nonzero(layer[i, :, :, :])) == 0:
                        output.append(0)
                    else:
                        output.append(1)
            else:
                # input
                input = []
                for i in range(layer.shape[1]):
                    if len(torch.nonzero(layer[:, i])) == 0:
                        input.append(0)
                    else:
                        input.append(1)
                # output
                output = []
                for i in range(layer.shape[0]):
                    if len(torch.nonzero(layer[i, :])) == 0:
                        output.append(0)
                    else:
                        output.append(1)
            # indices
            indices[(idx, id)] = torch.tensor(input)
            idx += 1
            indices[(idx, id)] = torch.tensor(output)
            idx += 1
        old_key = ()
        old_length = 0
        input = True
        for key, value in indices.items():
            length = len(value)
            # TODO: Handle early in training by resetting the optimizer
            if input == True:
                # breakpoint()
                if length == old_length:
                    indices[old_key] = value.__or__(indices[old_key])
                    indices[key] = value.__or__(indices[old_key])
                elif old_length != 0 and length % old_length == 0 and ('fc' in key[1] or 'classifier' in key[1]):
                    ratio = length // old_length
                    new_indices = torch.repeat_interleave(indices[old_key], ratio)
                    for i in range(old_length):
                        if sum(new_indices[i*ratio:ratio*(i+1)].__or__(value[i*ratio:ratio*(i+1)])) == ratio:
                            indices[old_key][i] = 1
                        else:
                            indices[old_key][i] = 0
                    indices[key] = torch.repeat_interleave(indices[old_key], ratio)
            old_length = length
            old_key = key
            input = not input
        self.structured_prune(indices)
        return indices

    def grow(self, percentage, train_loader):
        device = self.model.device
        iterations = SNIP_BATCH_ITERATIONS
        net = self.model.eval()

        # accumalate gradients of multiple batches
        net.zero_grad()
        loss_sum = torch.zeros([1]).to(self.device)
        for i, (x, y) in enumerate(train_loader):

            if i == iterations: break

            inputs = x.to(self.model.device)
            targets = y.to(self.model.device)
            outputs = net.forward(inputs)
            loss = F.nll_loss(outputs, targets) / iterations
            loss.backward()
            loss_sum += loss.item()
        # get elasticities
        grads_abs = {}
        for name, layer in net.named_modules():
            if "Norm" in str(layer): continue
            if name + ".weight" in self.model.mask:
                grads_abs[name + ".weight"] = torch.abs(layer.weight.grad)
        all_scores = torch.cat([torch.flatten(x) for _, x in grads_abs.items()])
        # percentage + the number of elements that are already there
        num_params_to_grow = int(
            len(all_scores) * (percentage) + sum([len(torch.nonzero(t)) for t in self.model.mask.values()]))
        if num_params_to_grow < 1:
            num_params_to_grow += 1
        elif num_params_to_grow > len(all_scores):
            num_params_to_grow = len(all_scores)

        # threshold
        threshold, _ = torch.topk(all_scores, num_params_to_grow, sorted=True)
        acceptable_score = threshold[-1]
        # grow
        for name, grad in grads_abs.items():
            self.model.mask[name] = ((grad) > acceptable_score).__or__(
                self.model.mask[name].bool()).float().to(self.device)

        self.model.apply_weight_mask()

    def get_weight_saliencies(self, train_loader, ood_loader=None):

        device = self.model.device

        iterations = SNIP_BATCH_ITERATIONS

        net = self.model.eval()

        self.their_implementation(device, iterations, net, train_loader)

        # collect gradients
        grads = {}
        for name, layer in net.named_modules():
            if "Norm" in str(layer): continue
            if name + ".weight" in self.model.mask:
                grads[name + ".weight"] = torch.abs(
                    # grads[name + ".weight"] = -(
                    layer.weight.data * layer.weight.grad)

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for _, x in grads.items()])

        so = all_scores.sort().values

        norm_factor = 1
        log10 = so.log10()
        all_scores.div_(norm_factor)

        self.model = self.model.train()
        self.model.zero_grad()

        return all_scores, grads, log10, norm_factor

    def their_implementation(self, device, iterations, net, train_loader):
        net.zero_grad()
        weights = []
        for name, layer in net.named_modules():
            if (name + ".weight") in net.mask:
                weights.append(layer.weight)
        inputs_one = []
        targets_one = []
        grad_w = None
        grad_f = None
        for w in weights:
            w.requires_grad_(True)
        dataloader_iter = iter(train_loader)
        for it in range(iterations):
            inputs, targets = next(dataloader_iter)
            N = inputs.shape[0]
            din = copy.deepcopy(inputs)
            dtarget = copy.deepcopy(targets)

            start = 0
            intv = 20

            while start < N:
                end = min(start + intv, N)
                inputs_one.append(din[start:end])
                targets_one.append(dtarget[start:end])
                if not self.generative:
                    outputs = net.forward(inputs[start:end].to(device))  # divide by temperature to make it uniform
                    loss = F.cross_entropy(outputs, targets[start:end].to(device))
                else:
                    log_p, logdet, _ = net(inputs[start:end].to(device))
                    logdet = logdet.mean()
                    loss, _, _ = self.loss_f(log_p, logdet, self.img_size, self.nbins, channels=self.channels)
                grad_w_p = autograd.grad(loss, weights, create_graph=False)
                # grad_w_p = autograd.grad(outputs, weights, grad_outputs=torch.ones_like(outputs), create_graph=False)
                if grad_w is None:
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]
                start = end
        for it in range(len(inputs_one)):
            inputs = inputs_one.pop(0).to(device)
            targets = targets_one.pop(0).to(device)
            if not self.generative:
                outputs = net.forward(inputs)  # divide by temperature to make it uniform
                loss = F.cross_entropy(outputs, targets)
            else:
                log_p, logdet, _ = net(inputs)
                logdet = logdet.mean()
                loss, _, _ = self.loss_f(log_p, logdet, self.img_size, self.nbins, channels=self.channels)
            grad_f = autograd.grad(loss, weights, create_graph=True)
            # grad_f = autograd.grad(outputs, weights, grad_outputs=torch.ones_like(outputs), create_graph=True)
            z = 0
            count = 0
            for name, layer in net.named_modules():
                if (name + ".weight") in net.mask:
                    z += (grad_w[count] * grad_f[count] * net.mask[name + ".weight"]).sum()
                    count += 1
            z.backward()

    def structured_prune(self, indices):
        summed_weights = sum([np.prod(x.shape) for name, x in self.model.named_parameters() if "weight" in name])
        # handle outer layers
        if not self.model._outer_layer_pruning:
            offsets = [len(x[0][1]) for x in lookahead_finished(indices.items()) if x[1][0] or x[1][1]]
            # breakpoint()
        #     all_scores = all_scores[offsets[0]:-offsets[1]]
        # prune
        summed_pruned = 0
        toggle_row_column = True
        cutoff = 0
        length_nonzero = 0
        for ((identification, name), grad), (first, last) in lookahead_finished(indices.items()):
            # breakpoint()
            binary_keep_neuron_vector = ((grad) > 0).float().to(self.device)
            corresponding_weight_parameter = [val for key, val in self.model.named_parameters() if key == name][0]
            is_conv = len(corresponding_weight_parameter.shape) > 2
            corresponding_module: nn.Module = \
                [val for key, val in self.model.named_modules() if key == name.split(".weight")[0]][0]

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
                                                          corresponding_weight_parameter)
            else:

                cutoff, length_nonzero = self.handle_middle_layers(binary_keep_neuron_vector,
                                                                   cutoff,
                                                                   is_conv,
                                                                   length_nonzero,
                                                                   corresponding_module,
                                                                   name,
                                                                   toggle_row_column,
                                                                   corresponding_weight_parameter)

            cutoff, summed_pruned = self.print_layer_progress(cutoff,
                                                              indices,
                                                              length_nonzero,
                                                              name,
                                                              summed_pruned,
                                                              toggle_row_column,
                                                              corresponding_weight_parameter)
            toggle_row_column = not toggle_row_column
        for line in str(self.model).split("\n"):
            if "BatchNorm" in line or "Conv" in line or "Linear" in line or "AdaptiveAvg" in line or "Sequential" in line:
                print(line)
        print("final percentage after snap:", summed_pruned / summed_weights)

        self.model.apply_weight_mask()

    def handle_middle_layers(self,
                             binary_vector,
                             cutoff,
                             is_conv,
                             length_nonzero,
                             module,
                             name,
                             toggle_row_column,
                             weight):

        indices = binary_vector.bool()
        length_nonzero_before = int(np.prod(weight.shape))
        n_remaining = binary_vector.sum().item()
        if not toggle_row_column:
            self.handle_output(indices,
                               is_conv,
                               module,
                               n_remaining,
                               name,
                               weight)

        else:
            cutoff, length_nonzero = self.handle_input(cutoff,
                                                       indices,
                                                       is_conv,
                                                       length_nonzero,
                                                       module,
                                                       n_remaining,
                                                       name,
                                                       weight)

        cutoff += (length_nonzero_before - int(np.prod(weight.shape)))
        return cutoff, length_nonzero

    def handle_input(self, cutoff, indices, is_conv, length_nonzero, module, n_remaining, name, weight):
        """ shrinks a input dimension """
        module.update_input_dim(n_remaining)
        length_nonzero = int(np.prod(weight.shape))
        cutoff = 0
        if is_conv:
            weight.data = weight[:, indices, :, :]
            try:
                weight.grad.data = weight.grad.data[:, indices, :, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][:, indices, :, :]
        else:
            if ((indices.shape[0] % weight.shape[0]) == 0) and not (weight.shape[1] == indices.shape[0]):
                ratio = weight.shape[1] // indices.shape[0]
                module.update_input_dim(n_remaining * ratio)
                new_indices = torch.repeat_interleave(indices, ratio)
                weight.data = weight[:, new_indices]
                if name in self.model.mask:
                    self.model.mask[name] = self.model.mask[name][:, new_indices]
                try:
                    weight.grad.data = weight.grad.data[:, new_indices]
                except AttributeError:
                    pass
            else:
                weight.data = weight[:, indices]
                try:
                    weight.grad.data = weight.grad.data[:, indices]
                except AttributeError:
                    pass
                if name in self.model.mask:
                    self.model.mask[name] = self.model.mask[name][:, indices]
        if self.model.is_tracking_weights:
            raise NotImplementedError
        return cutoff, length_nonzero

    def handle_output(self, indices, is_conv, module, n_remaining, name, weight):
        """ shrinks a output dimension """
        module.update_output_dim(n_remaining)
        self.handle_batch_norm(indices, n_remaining, name)
        if is_conv:
            weight.data = weight[indices, :, :, :]
            try:
                weight.grad.data = weight.grad.data[indices, :, :, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][indices, :, :, :]
        else:
            weight.data = weight[indices, :]
            try:
                weight.grad.data = weight.grad.data[indices, :]
            except AttributeError:
                pass
            if name in self.model.mask:
                self.model.mask[name] = self.model.mask[name][indices, :]
        self.handle_bias(indices, name)
        if self.model.is_tracking_weights:
            raise NotImplementedError

    def handle_bias(self, indices, name):
        """ shrinks a bias """
        bias = [val for key, val in self.model.named_parameters() if key == name.split("weight")[0] + "bias"][0]
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
                            param):

        n_remaining = binary_vector.sum().item()
        if first:
            length_nonzero = int(np.prod(param.shape))
            if self.model._outer_layer_pruning:
                module.update_input_dim(n_remaining)
                if is_conv:
                    permutation = (0, 3, 2, 1)
                    self.model.mask[name] = (self.model.mask[name].permute(permutation) * binary_vector).permute(
                        permutation)
                else:
                    self.model.mask[name] *= binary_vector
        elif last and self.model._outer_layer_pruning:
            module.update_output_dim(n_remaining)
            if is_conv:
                permutation = (3, 1, 2, 0)
                self.model.mask[name] = (self.model.mask[name].permute(permutation) * binary_vector).permute(
                    permutation)
            else:
                self.model.mask[name] = (binary_vector * self.model.mask[name].t()).t()
        if self.model._outer_layer_pruning:
            number_removed = (self.model.mask[name] == 0).sum().item()
            print("set to zero but not removed because of input-output compatibility:", number_removed,
                  f"({len(binary_vector) - n_remaining} features)")
        return length_nonzero

    def print_layer_progress(self, cutoff, grads_abs, length_nonzero, name, summed_pruned, toggle, weight):
        if not toggle:
            if len(grads_abs) == 2:
                cutoff /= 2
            summed_pruned += cutoff
            print("pruning", cutoff, "percentage", cutoff / length_nonzero, "length_nonzero", length_nonzero)
        return cutoff, summed_pruned
