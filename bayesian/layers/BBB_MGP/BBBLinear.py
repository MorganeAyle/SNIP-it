import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from metrics import calculate_kl as KL_DIV
from ..misc import ModuleWrapper2, Gaussian, ScaleMixtureGaussian


class BBBLinear(ModuleWrapper2):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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

        self.W_mu = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = Parameter(torch.empty((out_features, in_features), device=self.device))

        self.weight = Gaussian(self.W_mu, self.W_rho)

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features), device=self.device))

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

        return F.linear(input, weight, bias)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
