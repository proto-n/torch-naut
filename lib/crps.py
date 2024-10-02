from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
from . import utils
import numpy as np

def prepare_training(net, X_train, y_train):
    device = next(net.parameters()).device
    Xscaler = StandardScaler()
    yscaler = StandardScaler()
    X_train_s = torch.tensor(Xscaler.fit_transform(X_train), device=device, dtype=torch.float)
    y_train_s = torch.tensor(yscaler.fit_transform(y_train.reshape(-1,1)), device=device, dtype=torch.float)
    return X_train_s, y_train_s, (Xscaler, yscaler)

class EpsilonSampler(nn.Module):
    def __init__(self, n_dim):
        super(EpsilonSampler, self).__init__()
        self.n_dim = n_dim

    def forward(self, x, n_samples=10):
        eps = torch.randn(*x.shape[:-1], n_samples, self.n_dim, device=x.device)
        return torch.concatenate([x.unsqueeze(-2).expand(*([-1]*(len(x.shape)-1)), n_samples, -1), eps], dim=-1)

def crps_loss(yps, y):
    ml=yps.shape[-1]
    mrank = torch.argsort(torch.argsort(yps, dim=-1), dim=-1)
    return ((2/(ml*(ml-1)))*(yps-y)*(((ml-1)*(y<yps))-mrank)).sum(axis=-1)

def crps_loss_weighted(yps, w, y):
    ml = yps.shape[-1]
    sort_ix = torch.argsort(yps, dim=-1)
    sort_ix_reverse = torch.argsort(sort_ix)
    s = torch.take_along_dim(torch.cumsum(torch.take_along_dim(w, sort_ix, dim=-1), dim=-1), sort_ix_reverse, dim=-1)
    W = w.sum(dim=-1, keepdim=True)
    return (2/(ml*(ml-1))) * ( (w*(yps-y)*( (ml-1)*(y < yps) - s + (W-ml+w+1)/2) ) ).sum(dim=-1)

def crps_loss_mv(yps, y):
    return (yps-y.unsqueeze(-2)).norm(dim=-1).mean(dim=-1) - (1/2)*(yps.unsqueeze(-2)-yps.unsqueeze(-3)).norm(dim=-1).mean(dim=-1).sum(dim=-1)/(yps.shape[-2]-1)

def crps_loss_mv_weighted(yps, w, y):
    t1 = ((yps-y).norm(dim=-1)*w).mean(axis=-1)
    t2 = (((yps.unsqueeze(-2)-yps.unsqueeze(-3)).norm(dim=-1)*((w.unsqueeze(-1)*w.unsqueeze(-2)))).mean(dim=-1).sum(dim=-1)/(yps.shape[-2]-1))
    return t1 - (1/2)*t2

def predict_simple(net, scalers, X, n_samples=10, batch_size=32, runs=1):
    device = next(net.parameters()).device
    (Xscaler, yscaler) = scalers
    X_s = torch.tensor(Xscaler.transform(X), device=device, dtype=torch.float)
    preds = []
    with torch.no_grad():
        for ixs in utils.get_batch_ixs(X_s, batch_size):
            preds.append(net(X_s[ixs], n_samples=n_samples).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return yscaler.inverse_transform(preds.reshape(preds.shape[0], -1))

def predict(net, scalers, X, n_samples=10, batch_size=32, runs=1):
    device = next(net.parameters()).device
    (Xscaler, yscaler) = scalers
    X_s = torch.tensor(Xscaler.transform(X), device=device, dtype=torch.float)
    preds = []
    weights = []
    with torch.no_grad():
        for ixs in utils.get_batch_ixs(X_s, batch_size):
            xs = X_s[ixs]
            pred_, weight_ = zip(*[net(xs, n_samples=n_samples) for i in range(runs)])
            pred, weight = torch.concat(pred_, dim=-1), torch.concat(weight_, dim=-1)
            preds.append(pred.cpu().numpy())
            weights.append(weight.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    weights = np.concatenate(weights, axis=0)
    return yscaler.inverse_transform(preds.reshape(preds.shape[0], -1)), weights.reshape(weights.shape[0], -1)

def nll_gpu(pred_samples, y):
    if(torch.is_tensor(pred_samples)):
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
        t_std_X = (tkernels - t_mean)/t_sqcov
        _glob_bw = np.power(tkernels.shape[1] * (3/4), -1/5) # silverman for univariate
        t_invbw = torch.ones(1, 1, tkernels.shape[1]) / _glob_bw
        t_norm = t_invbw / (t_sqcov.unsqueeze(2)*np.sqrt(2*np.pi)) / tkernels.shape[1]
        t_kde_values = torch.sum(torch.exp(-0.5*t_invbw**2*(t_std_X.unsqueeze(1)-t_std_X.unsqueeze(2))**2)*t_norm, dim=2)
        t_g = (torch.exp(torch.sum(torch.log(t_kde_values), dim=1) / tkernels.shape[1])).unsqueeze(1)
        t_inv_loc_bw = (t_kde_values / t_g)**(0.5)

        # predict
        t_p_ = (tpoint-t_mean)/t_sqcov
        t_invbw = torch.ones(1, 1, tkernels.shape[1]) * t_inv_loc_bw / _glob_bw
        t_norm = t_invbw / (t_sqcov.unsqueeze(0)*np.sqrt(2*np.pi)) / tkernels.shape[1]
        likelihoods_all.append(torch.sum(torch.exp(-0.5*t_invbw**2*((t_std_X-t_p_)**2)).squeeze()*t_norm, dim=2).squeeze())
    return -torch.log(torch.cat(likelihoods_all, dim=0))

def nll_gpu_weighted(pred_samples, pred_weights, y, max_pilot_samples=None, batch_size=16):
    if(torch.is_tensor(pred_samples)):
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
        t_std_X = (tkernels - t_mean)/t_sqcov
        _glob_bw = np.power(pilot_samples_num * (3/4), -1/5) # silverman for univariate
        t_invbw = torch.ones(1, 1, 1) / _glob_bw
        t_norm = t_invbw / (t_sqcov.unsqueeze(2)*np.sqrt(2*np.pi)) / pilot_samples_num
        if not limit_pilot_samples:
            t_std_X_pilot = t_std_X
            self_kde_correction = 0
        else:
            pilot_ranks = ((torch.arange(pilot_samples_num)/pilot_samples_num)*tkernels.shape[-1]).int().to(tkernels.device)
            tkernels_sort_ix = torch.argsort(tkernels, dim=-1)
            pilot_indices = tkernels_sort_ix[:, pilot_ranks]
            t_std_X_pilot = torch.take_along_dim(t_std_X, pilot_indices, dim=-1)
            self_kde_correction = 1
        t_kde_values = torch.sum((
            torch.exp(-0.5*t_invbw**2*(t_std_X_pilot.unsqueeze(1)-t_std_X.unsqueeze(2))**2)+self_kde_correction
        )*t_norm, dim=2)
        t_g = (torch.exp(torch.sum(torch.log(t_kde_values), dim=1) / tkernels.shape[1])).unsqueeze(1)
        t_inv_loc_bw = (t_kde_values / t_g)**(0.5)

        # evaluate points
        
        if limit_pilot_samples:
            _glob_bw = np.power(tkernels.shape[1] * (3/4), -1/5) # silverman for univariate
        t_p_ = (tpoint-t_mean)/t_sqcov
        t_invbw = torch.ones(1, tkernels.shape[1]) * t_inv_loc_bw / _glob_bw
        t_norm_log = t_invbw.log() - (t_sqcov*np.sqrt(2*np.pi)).log() + tweights.log() - tweights.sum(dim=1, keepdim=True).log()
        loglikelihoods_all.append(torch.logsumexp(-0.5*t_invbw**2*((t_std_X-t_p_)**2)+t_norm_log, dim=1))
    return -torch.cat(loglikelihoods_all, dim=0)