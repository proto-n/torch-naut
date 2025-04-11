import torch
from torch import nn
from typing import Optional


class EpsilonSampler(nn.Module):
    """Layer that adds random normal samples to enable probabilistic predictions.

    This layer transforms input tensors by concatenating random samples from a standard
    normal distribution. The number of samples can be controlled globally using the context
    manager interface or per-call using the n_samples parameter.

    Args:
        sample_dim (int): Number of random dimensions to add

    Example:
        >>> sampler = EpsilonSampler(16)
        >>> # Default number of samples (100)
        >>> out = sampler(x)  # Shape: [batch, 100, features+16]
        >>>
        >>> # Override samples for a specific call
        >>> out = sampler(x, n_samples=1000)  # Shape: [batch, 1000, features+16]
        >>>
        >>> # Use context manager to temporarily change default samples
        >>> with EpsilonSampler.n_samples(500):
        ...     out = sampler(x)  # Shape: [batch, 500, features+16]
        >>> out = sampler(x)  # Back to default 100 samples
    """

    _global_n_samples: Optional[int] = (
        None  # Static attribute for context-manager n_samples
    )

    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim

    def forward(self, x, n_samples=None):
        """Forward pass adding random normal samples.

        Args:
            x (torch.Tensor): Input tensor
            n_samples (int, optional): Override number of samples for this call.
                                     If None, uses the current default value.
        """
        if n_samples is None:
            if EpsilonSampler._global_n_samples is not None:
                n_samples = EpsilonSampler._global_n_samples
            else:
                n_samples = 100

        eps = torch.randn(*x.shape[:-1], n_samples, self.n_dim, device=x.device)
        return torch.concatenate(
            [x.unsqueeze(-2).expand(*([-1] * (len(x.shape) - 1)), n_samples, -1), eps],
            dim=-1,
        )

    class _OverrideNSamples:  # Private inner class
        def __init__(self, n_samples, force=False):
            self.n_samples = n_samples
            self.force = force

        def __enter__(self):
            self.original_n_samples = EpsilonSampler._global_n_samples
            if self.force:
                EpsilonSampler._global_n_samples = self.n_samples
            else:
                # Only the first override is used
                if EpsilonSampler._global_n_samples is None:
                    EpsilonSampler._global_n_samples = self.n_samples
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            EpsilonSampler._global_n_samples = self.original_n_samples

    @classmethod
    def n_samples(cls, n_samples, force=False):
        return cls._OverrideNSamples(n_samples, force=force)

    @classmethod
    def get_n_samples(cls, default=None):
        if default is None:
            return EpsilonSampler._global_n_samples
        else:
            return (
                default
                if EpsilonSampler._global_n_samples is None
                else EpsilonSampler._global_n_samples
            )


def crps_loss(yps, y):
    """Calculates the Continuous Ranked Probability Score (CRPS) loss.

    Args:
        yps: Tensor of predicted samples [batch x num_samples]
        y: Target values [batch x 1]

    Returns:
        CRPS loss value per batch element
    """
    ml = yps.shape[-1]
    mrank = torch.argsort(torch.argsort(yps, dim=-1), dim=-1)
    return ((2 / (ml * (ml - 1))) * (yps - y) * (((ml - 1) * (y < yps)) - mrank)).sum(
        axis=-1
    )


def crps_loss_weighted(yps, w, y):
    """Calculates the weighted Continuous Ranked Probability Score (CRPS) loss.

    Args:
        yps: Tensor of predicted samples [batch x num_samples]
        w: Sample weights [batch x num_samples]
        y: Target values [batch x 1]

    Returns:
        Weighted CRPS loss value per batch element
    """
    ml = yps.shape[-1]
    sort_ix = torch.argsort(yps, dim=-1)
    sort_ix_reverse = torch.argsort(sort_ix)
    s = torch.take_along_dim(
        torch.cumsum(torch.take_along_dim(w, sort_ix, dim=-1), dim=-1),
        sort_ix_reverse,
        dim=-1,
    )
    W = w.sum(dim=-1, keepdim=True)
    return (2 / (ml * (ml - 1))) * (
        w * (yps - y) * ((ml - 1) * (y < yps) - s + (W - ml + w + 1) / 2)
    ).sum(dim=-1)


def crps_loss_mv(yps, y):
    """Calculates the multivariate CRPS (Energy Score) loss.

    Args:
        yps: Tensor of predicted samples [batch x num_samples x dims]
        y: Target values [batch x dims]

    Returns:
        Multivariate CRPS (Energy Score) loss value per batch element
    """
    return (yps - y.unsqueeze(-2)).norm(dim=-1).mean(dim=-1) - (1 / 2) * (
        yps.unsqueeze(-2) - yps.unsqueeze(-3)
    ).norm(dim=-1).mean(dim=-1).sum(dim=-1) / (yps.shape[-2] - 1)


def crps_loss_mv_weighted(yps, w, y):
    """Calculates the weighted multivariate CRPS (Energy Score) loss.

    Args:
        yps: Tensor of predicted samples [batch x num_samples x dims]
        w: Sample weights [batch x num_samples]
        y: Target values [batch x dims]

    Returns:
        Weighted multivariate CRPS (Energy Score) loss value per batch element
    """
    t1 = ((yps - y.unsqueeze(-2)).norm(dim=-1) * w).mean(axis=-1)
    t2 = (
        (yps.unsqueeze(-2) - yps.unsqueeze(-3)).norm(dim=-1)
        * (w.unsqueeze(-1) * w.unsqueeze(-2))
    ).mean(dim=-1).sum(dim=-1) / (yps.shape[-2] - 1)
    return t1 - (1 / 2) * t2
