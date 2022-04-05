import torch.nn.functional as F
import torch.autograd as autograd
import torch


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


class GetSubnetFaster(autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k_val = percentile(scores, sparsity * 100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


def linear_forward(layer, x):
    # subnet = GetSubnet.apply(layer.gov.float(), layer.prune_rate)
    subnet = GetSubnetFaster.apply(layer.gov.float(), torch.zeros_like(layer.gov.data), torch.ones_like(layer.gov.data), layer.prune_rate)
    w = layer.weight.data * subnet
    return F.linear(x.float(),
                    w,
                    # self.gov.float(),
                    bias=layer.bias.float())


def conv_forward(layer, x):
    # subnet = GetSubnet.apply(layer.gov.float(), layer.prune_rate)
    subnet = GetSubnetFaster.apply(layer.gov.float(), torch.zeros_like(layer.gov.data), torch.ones_like(layer.gov.data), layer.prune_rate)
    w = layer.weight.data * subnet
    return (F.conv2d(x,
                     w,
                     # self.gov.float(),
                     layer.bias,
                     layer.stride,
                     layer.padding,
                     layer.dilation,
                     layer.groups))


def calculate_fan_in(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size

    return fan_in
