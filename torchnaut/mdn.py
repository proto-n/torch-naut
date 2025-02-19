import numpy as np
import torch
from torch import nn


class MDN(nn.Module):
    """Univariate Mixture Density Network utility class.

    Expected network output shape: [batch x num_components x 3]
    - Output[:, :, 0]: means (mu)
    - Output[:, :, 1]: standard deviations (sigma)
    - Output[:, :, 2]: mixture weights (pi)
    """

    def __init__(self, n_components):
        """Initialize MDN utility class.

        Args:
            n_components: Number of mixture components
        """
        super(MDN, self).__init__()
        self.n_components = n_components

    def get_dist(self, p):
        """Convert network output to mixture distribution object.

        Args:
            p: Input tensor [batch x num_components x 3] containing (mu, sigma, pi)

        Returns:
            PyTorch mixture distribution object
        """
        loc = p[:, :, 0]
        scale = nn.functional.softplus(torch.clamp(p[:, :, 1], min=-15))
        mixture_weight_dist = torch.distributions.Categorical(
            logits=torch.clamp(p[:, :, 2], min=-15, max=15)
        )
        component_dist = torch.distributions.Normal(loc=loc, scale=scale)
        return torch.distributions.mixture_same_family.MixtureSameFamily(
            mixture_weight_dist, component_dist
        )

    def log_likelihood(self, p, y, min_log_proba=-np.inf):
        """Calculate log likelihood of mixture density network output.

        Args:
            p: Transformed output tensor
            y: Target values
            min_log_proba: Minimum log probability for clamping

        Returns:
            Log likelihood values per batch element
        """
        mixture_dist = self.get_dist(p)
        return mixture_dist.log_prob(y).clamp(min=min_log_proba)

    def expected_value(self, p):
        """Calculate expected value of the mixture distribution.

        Args:
            p: Transformed output tensor

        Returns:
            Expected value per batch element
        """
        mixture_dist = self.get_dist(p)
        return mixture_dist.mean

    def sample(self, p, n=100):
        """Draw samples from the mixture distribution.

        Args:
            p: Transformed output tensor
            n: Number of samples to draw

        Returns:
            n samples from the mixture distribution
        """
        mixture_dist = self.get_dist(p)
        return mixture_dist.sample((n,)).T


class MDNMV(nn.Module):
    """Multivariate Mixture Density Network utility class.

    Expected network output shape: [batch x num_components x (1 + n_outputs + tril_params)]
    - Output[:, :, 0]: mixture weights (pi)
    - Output[:, :, 1:n_outputs+1]: means (mu)
    - Output[:, :, n_outputs+1:]: lower triangular elements of scale matrix
    """

    def __init__(self, n_components, n_outputs):
        """Initialize multivariate MDN utility class.

        Args:
            n_components: Number of mixture components
            n_outputs: Dimensionality of the output space
        """
        super(MDNMV, self).__init__()
        self.n_components = n_components
        self.n_outputs = n_outputs
        # Number of parameters per component (mean + lower triangular elements)
        self.n_params_per_comp = n_outputs + (n_outputs * (n_outputs + 1) // 2)

        # Template for reconstructing lower triangular matrix
        self.register_buffer(
            "tril_template",
            torch.zeros(self.n_outputs, self.n_outputs, dtype=torch.int64),
        )
        tril_ix = torch.tril_indices(self.n_outputs, self.n_outputs)
        self.tril_template[tril_ix.tolist()] = torch.arange(tril_ix.shape[1])

    def get_dist(self, p):
        """Convert network output to mixture distribution object.
        Also handles all necessary transformations (activations and clamping).

        Args:
            p: Network output tensor containing mixture parameters

        Returns:
            PyTorch mixture distribution object
        """
        x = p.view(p.shape[0], self.n_components, 1 + self.n_params_per_comp)

        pi = x[:, :, 0]
        loc = x[:, :, 1 : self.n_outputs + 1]
        st_par = x[:, :, self.n_outputs + 1 :]

        scale_trils_raw = torch.tril(
            torch.gather(
                st_par.unsqueeze(-2).expand(-1, -1, self.tril_template.shape[0], -1),
                -1,
                self.tril_template.unsqueeze(0)
                .unsqueeze(0)
                .expand(st_par.shape[0], st_par.shape[1], -1, -1),
            )
        )
        diag_activated = torch.nn.functional.softplus(
            torch.diagonal(scale_trils_raw, dim1=-2, dim2=-1).clamp(min=-15)
        )
        scale_trils = torch.diagonal_scatter(
            scale_trils_raw, diag_activated, dim1=-2, dim2=-1
        )
        component_dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=loc,
            scale_tril=scale_trils,
        )
        mixture_weight_dist = torch.distributions.Categorical(logits=pi.clamp(min=-15))
        mixture_dist = torch.distributions.mixture_same_family.MixtureSameFamily(
            mixture_weight_dist, component_dist
        )
        return mixture_dist

    def log_likelihood(self, p, y, min_log_proba=-np.inf):
        """Calculate log likelihood of mixture density network output.

        Args:
            p: Network output tensor
            y: Target values
            min_log_proba: Minimum log probability for clamping

        Returns:
            Log likelihood values per batch element
        """
        mixture_dist = self.get_dist(p)
        return mixture_dist.log_prob(y).clamp(min_log_proba)

    def expected_value(self, p):
        """Calculate expected value of the mixture distribution.

        Args:
            p: Output tensor

        Returns:
            Expected value per batch element
        """
        mixture_dist = self.get_dist(p)
        return mixture_dist.mean

    def sample(self, p, n=100):
        """Draw samples from the mixture distribution.

        Args:
            p: Output tensor
            n: Number of samples to draw

        Returns:
            n samples from the distribution [batch x n x dims]
        """
        mixture_dist = self.get_dist(p)
        return mixture_dist.sample((n,)).permute(1, 0, 2)
