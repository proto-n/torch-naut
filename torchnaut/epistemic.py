import torch
from torch import nn
import numpy as np


def get_kl_term(net):
    """Calculate total KL divergence term for all Bayesian layers in network.

    Args:
        net: Neural network containing Bayesian layers

    Returns:
        Sum of KL divergence terms from all Bayesian layers
    """
    kl = 0
    for m in net.modules():
        if hasattr(m, "get_kl_term"):
            kl += m.get_kl_term()
    return kl


class BayesianParameter(nn.Module):
    """Bayesian parameter that uses the reparameterization trick for sampling.

    Maintains a variational posterior distribution over the parameter values
    and computes KL divergence against a specified prior distribution.

    Args:
        shape: Shape of the parameter tensor
        prior_mu: Mean of the prior normal distribution
        prior_sigma: Standard deviation of the prior normal distribution
    """

    def __init__(self, shape, prior_mu, prior_sigma):
        super().__init__()
        self.shape = shape
        self.mu = nn.Parameter(
            torch.ones(*shape) * prior_mu + torch.randn(*shape) / np.sqrt(shape[-1])
        )
        self.rho = nn.Parameter(
            torch.ones(*shape) * torch.log(torch.exp(torch.tensor(prior_sigma)) - 1)
        )
        self.register_buffer("prior_mu", torch.zeros_like(self.mu) + prior_mu)
        self.register_buffer("prior_sigma", torch.zeros_like(self.rho) + prior_sigma)

    def get_kl_term(self):
        return torch.distributions.kl.kl_divergence(
            torch.distributions.Normal(self.mu, nn.functional.softplus(self.rho)),
            torch.distributions.Normal(self.prior_mu, self.prior_sigma),
        ).sum()

    def forward(self):
        epsilon = torch.randn(*self.shape, device=self.mu.device)
        return self.mu + nn.functional.softplus(self.rho.clamp(min=-30)) * epsilon


class BayesianLinear(nn.Module):
    """Fully connected layer with Bayesian parameters.

    Uses variational inference with reparameterization to sample weights and biases
    from posterior distributions during forward passes.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        prior_mu: Mean of the prior normal distribution for weights
        prior_sigma: Standard deviation of the prior normal distribution for weights
    """

    def __init__(self, in_features, out_features, prior_mu=0, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = BayesianParameter(
            (in_features, out_features), prior_mu, prior_sigma
        )
        self.bias = BayesianParameter((out_features,), 0, prior_sigma)

    def forward(self, x):
        weight = self.weight()
        bias = self.bias()
        return x @ weight + bias


class CRPSEnsemble(nn.Module):
    """Ensemble of networks for CRPS computation.

    Wraps multiple networks and runs forward passes through all of them,
    concatenating the outputs along `concat_dim` for CRPS calculation.

    Args:
        networks: List of networks to include in the ensemble
    """

    def __init__(self, networks, concat_dim=-2):
        super().__init__()
        self.networks = nn.ModuleList(networks)
        self.concat_dim = concat_dim

    def forward(self, *args, **kwargs):
        # trickery to handle a variable number of outputs
        outputs = [network(*args, **kwargs) for network in self.networks]
        outputs = zip(*(o if isinstance(o, (tuple, list)) else (o,) for o in outputs))
        outputs_cat = [
            torch.cat([o.unsqueeze(self.concat_dim) for o in out], dim=self.concat_dim)
            for out in outputs
        ]
        if len(outputs_cat)==1:
            return outputs_cat[0]
        else:   
            return tuple(outputs_cat) 
