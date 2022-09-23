############### Configuration file for Bayesian ###############
layer_type = 'lrt'  # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 250
lr_start = 0.002
num_workers = 0
valid_size = 0.2
batch_size = 128
train_ens = 1
valid_ens = 1
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value

import torch
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

