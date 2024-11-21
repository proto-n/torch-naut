# %%
# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import lib.crps as crps
import lib.utils as utils
from copy import deepcopy


n_conv = 5
n_filters = 8
n_pool = 3
n_pixels_1 = 128
n_pixels_2 = 128
size_out_1 = 8
size_out_2 = 8
nrand = 200


class JointPositionExtractor(nn.Module):
    def __init__(self, J=14):
        super(JointPositionExtractor, self).__init__()
        
        input_size = size_out_1 * size_out_2 * n_filters + nrand
        self.layers = nn.ModuleList([
            nn.Conv2d(1, n_filters, n_conv, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(n_pool),
            nn.Conv2d(n_filters, n_filters, n_conv, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(n_pool),
            nn.Conv2d(n_filters, n_filters, n_conv, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Flatten(),
            crps.EpsilonSampler(nrand),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        ])
        self.samples = nn.Linear(1024, 3 * J)
        self.weights = nn.Linear(1024, 1)

    def forward(self, x, n_samples=20):
        for l in self.layers:
            if type(l)==crps.EpsilonSampler:
                x = l(x, n_samples=n_samples)
            else:
                x = l(x)

        samples = self.samples(x)
        weights_raw = self.weights(x).squeeze()
        weights_log_softmax = nn.functional.log_softmax(weights_raw, dim=-1)
        weights = torch.exp(weights_log_softmax+np.log(weights_log_softmax.shape[-1]))
        return samples, weights.squeeze(-1)
    

class Ensemble(nn.Module):
    def __init__(self, n_networks, *args, **kwargs):
        super(Ensemble, self).__init__()
        self.networks = nn.ModuleList([
            JointPositionExtractor(*args, **kwargs) for i in range(n_networks)
        ])

    def forward(self, *args, **kwargs):
        preds, weights = zip(*[network(*args, **kwargs) for network in self.networks])
        return torch.cat([p.unsqueeze(-3) for p in preds], dim=-3), torch.cat([w.unsqueeze(-2) for w in weights], dim=-2)


if __name__ == "__main__":
    device="cuda:"+str(utils.select_gpu_with_low_usage())
    print("Using device", device)

    X_train = np.load('./data/X_train.npy')
    y_train = np.load('./data/Y_train.npy')

    X_test = np.load('./data/X_test.npy')
    y_test = np.load('./data/Y_test.npy')


    class AxisStandardScaler:
        def __init__(self, axis=0):
            self.axis = axis
            self.mean_ = None
            self.std_ = None
        def fit_transform(self, arr):
            self.fit(arr)
            return self.transform(arr)
        def fit(self, arr):
            self.mean_ = np.mean(arr, axis=self.axis, keepdims=True)
            self.std_ = np.std(arr, axis=self.axis, keepdims=True)
            return self
        def transform(self, arr):
            return (arr - self.mean_) / self.std_
        def inverse_transform(self, arr_scaled):
            return arr_scaled * self.std_ + self.mean_


    def prepare_training(net, X_train, y_train):
        device = next(net.parameters()).device
        Xscaler = AxisStandardScaler(axis=0)
        yscaler = AxisStandardScaler(axis=0)
        X_train_s = torch.tensor(Xscaler.fit_transform(X_train), device=device, dtype=torch.float)
        y_train_s = torch.tensor(yscaler.fit_transform(y_train), device=device, dtype=torch.float)
        return X_train_s, y_train_s, (Xscaler, yscaler)

    def get_batch_ixs(like_tensor, batch_size=16, permute=False):
        if(like_tensor.shape[0] <= batch_size):
            return torch.arange(like_tensor.shape[0]).unsqueeze(0)
        if permute:
            ixs = torch.randperm(like_tensor.shape[0])
        else:
            ixs = torch.arange(like_tensor.shape[0])
        return torch.tensor_split(ixs, ixs.shape[0]//batch_size)

    lr=1e-5

    # %%
    net = Ensemble(5).to(device)
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=10)

    X_train_s, y_train_s, scalers = prepare_training(net, X_train, y_train)

    train_mask = np.random.rand(X_train_s.shape[0])>0.1
    X_train_sv = X_train_s[train_mask]
    y_train_sv = y_train_s[train_mask]
    X_val_sv = X_train_s[~train_mask]
    y_val_sv = y_train_s[~train_mask]

    max_patience = 20
    patience = 0
    best_val_loss = np.inf
    best_model = None

    for k in range(2000):
        losses = []
        for ixs in get_batch_ixs(X_train_sv, int(np.sqrt(X_train_sv.shape[0])), permute=True):
            X = X_train_sv[ixs]
            y = y_train_sv[ixs].flatten(1, -1)
            x, w = net(X)

            loss = crps.crps_loss_mv_weighted(x, w, y.unsqueeze(1).unsqueeze(1)).sum()

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            losses.append(loss.item())
        with torch.no_grad():
            val_losses = []
            for ixs in get_batch_ixs(X_val_sv, 1000):
                X = X_val_sv[ixs]
                y = y_val_sv[ixs].flatten(1, -1)
                x, w = net(X, n_samples=20)
                x_pool, w_pool = x.reshape(x.shape[0], -1, x.shape[-1]), w.reshape(w.shape[0], -1)
                loss = crps.crps_loss_mv_weighted(x_pool, w_pool, y.unsqueeze(1)).mean()
                val_losses.append(loss.item())
        scheduler.step()
        print(k, np.mean(losses), np.mean(val_losses))
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(net.state_dict())
            patience = 0
        else:
            patience += 1
        if patience > max_patience:
            break

    torch.save(best_model, "results/wcrps_ens_fix_handpose.pt")

    net.load_state_dict(best_model)

    Xscaler, yscaler = scalers

    X_test_s = torch.FloatTensor(Xscaler.transform(X_test)).to(device)

    with torch.no_grad():
        preds = []
        weights = []
        for ixs in get_batch_ixs(X_test_s, 200):
            X = X_test_s[ixs]
            x, w = net(X, n_samples=100)
            x_pool, w_pool = x.reshape(x.shape[0], -1, x.shape[-1]), w.reshape(w.shape[0], -1)
            preds.append(x_pool.detach().cpu().numpy())
            weights.append(w_pool.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    weights = np.concatenate(weights, axis=0)

    # %%
    preds = preds.reshape(preds.shape[0], preds.shape[1], *y_test.shape[1:])
    preds_rescaled = yscaler.inverse_transform(preds.reshape(-1, *y_test.shape[1:])).reshape(*preds.shape)

    pred_mean = preds_rescaled.mean(axis=1)

    mejee, mejee_std = np.linalg.norm((pred_mean-y_test), axis=-1).mean().mean(), np.linalg.norm((pred_mean-y_test), axis=-1).mean(axis=-1).std()
    majee, majee_std = np.linalg.norm((pred_mean-y_test), axis=-1).max(axis=-1).mean(), np.linalg.norm((pred_mean-y_test), axis=-1).max(axis=-1).std()
    ff = (np.linalg.norm((pred_mean-y_test), axis=-1)<0.8).all(axis=-1).mean()

    ixs_list = get_batch_ixs(X_test_s, 200)
    with torch.device(device):
        scores = []
        for ixs in tqdm(ixs_list):
            score = crps.crps_loss_mv_weighted(
                torch.FloatTensor(preds_rescaled[ixs]).flatten(2, -1),
                torch.FloatTensor(weights[ixs]),
                torch.FloatTensor(y_test[ixs]).unsqueeze(1).flatten(2, -1)
            )
            scores.append(score)

    probloss, probloss_std = torch.concatenate(scores).cpu().numpy().mean(), torch.concatenate(scores).cpu().numpy().std()

    # results needed in mm
    print("ProbLoss (mm):", "%.1f$\pm$%.3f"%(probloss*100, (probloss_std*100)/np.sqrt(X_test.shape[0])))
    print("MeJEE (mm):", "%.1f$\pm$%.3f"%(mejee*100, (mejee_std*100)/np.sqrt(X_test.shape[0])))
    print("MaJEE (mm):", "%.1f$\pm$%.3f"%(majee*100, (majee_std*100)/np.sqrt(X_test.shape[0])))
    print("FF (80mm):", "%.3f"%(ff*100))
