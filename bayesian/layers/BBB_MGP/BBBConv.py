import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import bayesian_utils as utils
from metrics import calculate_kl as KL_DIV
import config_bayesian as cfg
from ..misc import ModuleWrapper2, Gaussian, ScaleMixtureGaussian


class BBBConv2d(ModuleWrapper2):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, priors=None):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu_1': 0,
                'prior_mu_2': 0,
                'prior_sigma_1': math.exp(-0),
                'prior_sigma_2': math.exp(-6),
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
                'pi': 0.5
            }
        self.prior_mu_1 = priors['prior_mu_1']
        self.prior_mu_2 = priors['prior_mu_2']
        self.prior_sigma_1 = priors['prior_sigma_1']
        self.prior_sigma_2 = priors['prior_sigma_2']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.weight_prior = ScaleMixtureGaussian(priors['pi'], priors['prior_sigma_1'], priors['prior_sigma_2'],
                                                 priors['prior_mu_1'], priors['prior_mu_2'])
        self.bias_prior = ScaleMixtureGaussian(priors['pi'], priors['prior_sigma_1'], priors['prior_sigma_2'],
                                               priors['prior_mu_1'], priors['prior_mu_2'])

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.weight = Gaussian(self.W_mu, self.W_rho)

        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))

            self.bias = Gaussian(self.bias_mu, self.bias_rho)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

        self.mask = torch.ones_like(self.W_mu)

    def reset_parameters(self):
        self.W_mu.data.uniform_(*self.posterior_mu_initial)
        self.W_rho.data.uniform_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.uniform_(*self.posterior_mu_initial)
            self.bias_rho.data.uniform_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            weight = self.weight.sample()
            if self.use_bias:
                bias = self.bias.sample()
            else:
                bias = None
        else:
            weight = self.weight.mu
            bias = self.bias.mu if self.use_bias else None

        if self.training or sample:
            self.log_prior = self.weight_prior.log_prob(weight)
            self.log_variational_posterior = self.weight.log_prob(weight)
            if bias is not None:
                self.log_prior += self.bias_prior.log_prob(bias)
                self.log_variational_posterior += self.bias.log_prob(bias)

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
