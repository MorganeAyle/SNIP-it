import torch.nn.functional as F


def linear_forward(self, x):
    return F.linear(x.float(),
                    self.weight * self.gov.float(),
                    bias=self.bias.float())


def conv_forward(self, x):
    return (F.conv2d(x,
                     self.weight * self.gov.float(),
                     self.bias,
                     self.stride,
                     self.padding,
                     self.dilation,
                     self.groups))


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
