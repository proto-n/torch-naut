import torch
from . import utils
import numpy as np


def nll_gpu(pred_samples, y):
    """Calculate negative log likelihood using adaptive kernel density estimation.

    Uses Silverman's rule for global bandwidth and adaptive local bandwidth scaling.

    Args:
        pred_samples: Predicted samples [batch x num_samples]
        y: Target values [batch, 1]

    Returns:
        Negative log likelihood values per batch element
    """
    if torch.is_tensor(pred_samples):
        tkernels_all = pred_samples
        tpoint_all = y
    else:
        tkernels_all = torch.tensor(pred_samples).float()
        tpoint_all = torch.tensor(y).unsqueeze(1)

    likelihoods_all = []
    for ixs in utils.get_batch_ixs(tkernels_all):
        tkernels = tkernels_all[ixs]
        tpoint = tpoint_all[ixs]

        # fit
        t_sqcov = tkernels.std(dim=1, keepdim=True)
        t_mean = tkernels.mean(dim=1, keepdim=True)
        t_std_X = (tkernels - t_mean) / t_sqcov
        _glob_bw = np.power(
            tkernels.shape[1] * (3 / 4), -1 / 5
        )  # silverman for univariate
        t_invbw = torch.ones(1, 1, tkernels.shape[1]) / _glob_bw
        t_norm = (
            t_invbw / (t_sqcov.unsqueeze(2) * np.sqrt(2 * np.pi)) / tkernels.shape[1]
        )
        t_kde_values = torch.sum(
            torch.exp(
                -0.5 * t_invbw**2 * (t_std_X.unsqueeze(1) - t_std_X.unsqueeze(2)) ** 2
            )
            * t_norm,
            dim=2,
        )
        t_g = (
            torch.exp(torch.sum(torch.log(t_kde_values), dim=1) / tkernels.shape[1])
        ).unsqueeze(1)
        t_inv_loc_bw = (t_kde_values / t_g) ** (0.5)

        # predict
        t_p_ = (tpoint - t_mean) / t_sqcov
        t_invbw = torch.ones(1, 1, tkernels.shape[1]) * t_inv_loc_bw / _glob_bw
        t_norm = (
            t_invbw / (t_sqcov.unsqueeze(0) * np.sqrt(2 * np.pi)) / tkernels.shape[1]
        )
        likelihoods_all.append(
            torch.sum(
                torch.exp(-0.5 * t_invbw**2 * ((t_std_X - t_p_) ** 2)).squeeze()
                * t_norm,
                dim=2,
            ).squeeze()
        )
    return -torch.log(torch.cat(likelihoods_all, dim=0))


def nll_gpu_weighted(
    pred_samples, pred_weights, y, max_pilot_samples=None, batch_size=16
):
    """Calculate negative log likelihood using weighted adaptive kernel density estimation.

    Uses Silverman's rule for global bandwidth and adaptive local bandwidth scaling.

    Args:
        pred_samples: Predicted samples [batch x num_samples]
        pred_weights: Sample weights [batch x num_samples]
        y: Target values [batch, 1]
        max_pilot_samples: Maximum number of pilot samples for bandwidth estimation
        batch_size: Batch size for memory-efficient computation

    Returns:
        Negative log likelihood values per batch element
    """
    if torch.is_tensor(pred_samples):
        tkernels_all = pred_samples
        tpoint_all = y
        tweights_all = pred_weights
    else:
        tkernels_all = torch.tensor(pred_samples).float()
        tpoint_all = torch.tensor(y).unsqueeze(1)
        tweights_all = torch.tensor(pred_weights).float()

    loglikelihoods_all = []
    for ixs in utils.get_batch_ixs(tkernels_all, batch_size=batch_size):
        tkernels = tkernels_all[ixs]
        tpoint = tpoint_all[ixs]
        tweights = tweights_all[ixs]

        # find local bandwidth values

        pilot_samples_num = tkernels.shape[1]
        limit_pilot_samples = False
        if max_pilot_samples is not None and max_pilot_samples < pilot_samples_num:
            pilot_samples_num = max_pilot_samples
            limit_pilot_samples = True
        t_sqcov = tkernels.std(dim=1, keepdim=True)
        t_mean = tkernels.mean(dim=1, keepdim=True)
        t_std_X = (tkernels - t_mean) / t_sqcov
        _glob_bw = np.power(
            pilot_samples_num * (3 / 4), -1 / 5
        )  # silverman for univariate
        t_invbw = torch.ones(1, 1, 1) / _glob_bw
        t_norm = (
            t_invbw / (t_sqcov.unsqueeze(2) * np.sqrt(2 * np.pi)) / pilot_samples_num
        )
        if not limit_pilot_samples:
            t_std_X_pilot = t_std_X
            self_kde_correction = 0
        else:
            pilot_ranks = (
                (
                    (torch.arange(pilot_samples_num) / pilot_samples_num)
                    * tkernels.shape[-1]
                )
                .int()
                .to(tkernels.device)
            )
            tkernels_sort_ix = torch.argsort(tkernels, dim=-1)
            pilot_indices = tkernels_sort_ix[:, pilot_ranks]
            t_std_X_pilot = torch.take_along_dim(t_std_X, pilot_indices, dim=-1)
            self_kde_correction = 1
        t_kde_values = torch.sum(
            (
                torch.exp(
                    -0.5
                    * t_invbw**2
                    * (t_std_X_pilot.unsqueeze(1) - t_std_X.unsqueeze(2)) ** 2
                )
                + self_kde_correction
            )
            * t_norm,
            dim=2,
        )
        t_g = (
            torch.exp(torch.sum(torch.log(t_kde_values), dim=1) / tkernels.shape[1])
        ).unsqueeze(1)
        t_inv_loc_bw = (t_kde_values / t_g) ** (0.5)

        # evaluate points

        if limit_pilot_samples:
            _glob_bw = np.power(
                tkernels.shape[1] * (3 / 4), -1 / 5
            )  # silverman for univariate
        t_p_ = (tpoint - t_mean) / t_sqcov
        t_invbw = torch.ones(1, tkernels.shape[1]) * t_inv_loc_bw / _glob_bw
        t_norm_log = (
            t_invbw.log()
            - (t_sqcov * np.sqrt(2 * np.pi)).log()
            + tweights.log()
            - tweights.sum(dim=1, keepdim=True).log()
        )
        loglikelihoods_all.append(
            torch.logsumexp(
                -0.5 * t_invbw**2 * ((t_std_X - t_p_) ** 2) + t_norm_log, dim=1
            )
        )
    return -torch.cat(loglikelihoods_all, dim=0)
