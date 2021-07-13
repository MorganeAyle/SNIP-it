############### Configuration file for bayesian ###############
import math

layer_type = 'bbb'  # 'bbb' or 'lrt' or 'mgp
# layer_type = 'mgp'
activation_type = 'relu'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

# priors = {
#     'prior_mu_1': 0,
#     'prior_mu_2': 0,
#     'prior_sigma_1': math.exp(-0),
#     'prior_sigma_2': math.exp(-6),
#     'posterior_mu_initial': (-0.2, 0.2),
#     'posterior_rho_initial': (-5, -4),
#     'pi': 0.5
# }

n_epochs = 0
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256
train_ens = 1
valid_ens = 1
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value
