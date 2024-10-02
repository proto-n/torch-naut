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
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import lib.crps as crps
import lib.utils as utils


class CRPSModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CRPSModel, self).__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            crps.EpsilonSampler(64),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        ])
        self.samples = nn.Linear(128, output_dim)
        self.weights = nn.Linear(128, 1)

    def forward(self, x, n_samples=100):
        for l in self.layers:
            if type(l)==crps.EpsilonSampler:
                x = l(x, n_samples=n_samples)
            else:
                x = l(x)

        samples = self.samples(x)#.transpose(-1, -2).reshape(x.shape[0], self.nlayers, -1)
        weights_raw = self.weights(x)#.transpose(-1, -2).reshape(x.shape[0], self.nlayers, -1)
        weights_log_softmax = nn.functional.log_softmax(weights_raw, dim=-1)
        weights = torch.exp(weights_log_softmax+np.log(weights_log_softmax.shape[-1]))
        return samples, weights.squeeze(-1)
        
if __name__ == "__main__":
    device="cuda:"+str(utils.select_gpu_with_low_usage())

    x_samp_raw = np.random.uniform(-2, 2, size=2000000)
    y_samp_raw = np.random.uniform(-2, 2, size=2000000)
    norm_samp_raw = x_samp_raw**2+y_samp_raw**2
    circle_mask = norm_samp_raw<=2**2
    x_samp = x_samp_raw[circle_mask]
    y_samp = y_samp_raw[circle_mask]
    norm_samp = norm_samp_raw[circle_mask]

    train_X = torch.tensor(norm_samp, dtype=torch.float32).view(-1,1)
    train_y = torch.stack([
        torch.tensor(x_samp, dtype=torch.float32),
        torch.tensor(y_samp, dtype=torch.float32),
    ], dim=0).T

    max_epoch=1000
    lr=0.0001
    batch_size = 1000

    def prepare_training(net, X_train, y_train):
        device = next(net.parameters()).device
        Xscaler = StandardScaler()
        yscaler = StandardScaler()
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

    net = CRPSModel(1, 2).to(device)
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=10)

    X_train_s, y_train_s, scalers = prepare_training(net, train_X, train_y)

    for k in range(100):
        for ixs in tqdm(get_batch_ixs(X_train_s, batch_size, permute=True)):
            X = X_train_s[ixs]
            y = y_train_s[ixs]
            x, w = net(X)

            loss = crps.crps_loss_mv_weighted(x, w, y.unsqueeze(1)).sum()

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
        with torch.no_grad():
            samples, weights = net(torch.tensor([[0.25]]).float().to(device), n_samples=10000)
        plt.figure()
        plt.hist2d(*samples.squeeze().T.cpu().numpy(), weights=weights.squeeze().cpu().numpy(), bins=100, cmap='inferno')
        plt.gca().set_aspect(1)
        plt.savefig("results/crps_weighted/%d.png"%k)
        plt.show()
        plt.close()

        if k%10==9:
            fig, axs = plt.subplots(4, 4, figsize=(10, 10))
            (Xscaler, yscaler) = scalers
            for i, ax in zip(np.linspace(0, 2, 4*4+1)[1:]**2, axs.reshape(-1)):
                irt = Xscaler.transform([[i]]) # type: ignore
                with torch.no_grad():
                    samples, weights = net(torch.tensor(irt).float().to(device), n_samples=10000)
                sample = yscaler.inverse_transform(samples.squeeze().cpu().numpy())
                ax.hist2d(*sample.T, weights=weights.squeeze().cpu().numpy(), bins=100, cmap='inferno') # type: ignore
                ax.set_aspect('equal')
                ax.set_title("$f^{-1}(%.2f)$"%i)
                ax.set_xlim([-3, 3])
                ax.set_ylim([-3, 3])
                ax.set_facecolor('k')
            fig.tight_layout()
            plt.savefig("results/crps_weighted/%d_big.png"%k)
            plt.show()
            plt.close()
            torch.save(net.state_dict(), "results/crps_weighted/%d.pt"%k)
