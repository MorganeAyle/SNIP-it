import torch.nn.functional as F
import torch

"""
assisting maskeable functions for the methods we introduced.
"""

def snip_forward_conv2d(self, x):
    return F.conv2d(x,
                    self.weight * self.weight_mask,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups)


def snip_forward_linear(self, x):
    return F.linear(x.float(),
                    self.weight.float() * self.weight_mask.float(),
                    bias=self.bias)


def group_snip_forward_linear(self, x):
    return F.linear(x.float(),
                    self.weight,
                    bias=self.bias.float()) * self.gov_out.float()


def group_snip_conv2d_forward(self, x):
    return (F.conv2d(x,
                     self.weight,
                     self.bias,
                     self.stride,
                     self.padding,
                     self.dilation,
                     self.groups).permute(0, 3, 2, 1) * self.gov_out.float()).permute(0, 3, 2, 1)


def group_snip_invconv2d_forward(self, x):
    _, _, height, width = x.shape

    weight = self.calc_weight()

    out = (F.conv2d(x, weight).permute(0, 3, 2, 1) * self.gov_out.float()).permute(0, 3, 2, 1)
    logdet = height * width * torch.sum(self.w_s)

    return out, logdet
