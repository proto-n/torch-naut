import torch
from torch import nn
import numpy as np

def get_kl_term(net):
    kl = 0
    for m in net.modules():
        if hasattr(m, "get_kl_term"):
            kl += m.get_kl_term()
    return kl

class BayesianParameter(nn.Module):
    def __init__(self, shape, prior_mu, prior_sigma):
        super(BayesianParameter, self).__init__()
        self.shape = shape
        self.mu = nn.Parameter(torch.ones(*shape)*prior_mu + torch.randn(*shape)/np.sqrt(shape[-1]))
        self.rho = nn.Parameter(torch.ones(*shape)*torch.log(torch.exp(torch.tensor(prior_sigma))-1))
        self.register_buffer('prior_mu', torch.zeros_like(self.mu) + prior_mu)
        self.register_buffer('prior_sigma', torch.zeros_like(self.rho) + prior_sigma)

    def get_kl_term(self):
        return torch.distributions.kl.kl_divergence(
            torch.distributions.Normal(self.mu, nn.functional.softplus(self.rho)),
            torch.distributions.Normal(self.prior_mu, self.prior_sigma)
        ).sum()
    
    def forward(self):
        epsilon = torch.randn(*self.shape, device=self.mu.device)
        return self.mu + nn.functional.softplus(self.rho.clamp(min=-30)) * epsilon

class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0, prior_sigma=0.1):
        super(BayesianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = BayesianParameter((in_features, out_features), prior_mu, prior_sigma)
        self.bias = BayesianParameter((out_features,), 0, prior_sigma)

    def forward(self, x):
        weight = self.weight()
        bias = self.bias()
        return x @ weight + bias
    
