import torch
import torch.nn as nn

from models.Pruneable import Pruneable
from utils.constants import MIDDLE_POOL, PROD_MIDDLE_POOL


class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, basis_func):
        super(RBF, self).__init__()
        self.basis_func = basis_func

    def forward(self, input):
        return self.basis_func(input)


class RBFLayer(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        # print(input.shape)
        # print(self.centres.shape)
        # pdb.set_trace()
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        # print(x.shape)
        # print(c.shape)
        # pdb.set_trace()
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        # print(distances)
        # return self.basis_func(distances)
        return distances


def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


class LeNet5RBF(Pruneable):

    def __init__(self, device="cuda", output_dim=2, input_dim=(1, 1, 1), **kwargs):
        super(LeNet5RBF, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)

        channels, dim1, dim2 = input_dim

        leak = 0.05
        # gain = nn.init.calculate_gain('leaky_relu', leak)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            RBF(basis_func=gaussian),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            RBF(basis_func=gaussian),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            RBF(basis_func=gaussian),
        ).to(device)

        self.avgpool = nn.AdaptiveAvgPool2d(MIDDLE_POOL).to(device)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(120 * PROD_MIDDLE_POOL, 84, bias=True),
            nn.BatchNorm1d(84, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            RBF(basis_func=gaussian),

            nn.Dropout(p=0.3, inplace=False),
            RBFLayer(84, output_dim, gaussian),
            # self.Linear(84, output_dim, bias=True),
        ).to(device)

    def forward(self, x: torch.Tensor):
        x = self.conv.forward(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
