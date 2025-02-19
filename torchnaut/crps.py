import torch


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
