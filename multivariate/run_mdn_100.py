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
from copy import deepcopy
import lib.utils as utils

# utility class for multivariate MDN calculations
class MDNMV(nn.Module):
    def __init__(self, n_components, n_outputs):
        super(MDNMV, self).__init__()
        self.n_components = n_components
        self.n_outputs = n_outputs
        self.n_params_per_comp = n_outputs + (n_outputs * (n_outputs + 1) // 2)

        self.register_buffer('tril_template', torch.zeros(self.n_outputs, self.n_outputs, dtype=torch.int64))
        tril_ix = torch.tril_indices(self.n_outputs, self.n_outputs)
        self.tril_template[tril_ix.tolist()] = torch.arange(tril_ix.shape[1])

    def get_dist(self, p):
        x = p.view(p.shape[0], self.n_components, 1 + self.n_params_per_comp)

        pi = x[:, :, 0]
        loc = x[:, :, 1:self.n_outputs+1]
        st_par = x[:, :, self.n_outputs+1:]

        scale_trils_raw = torch.tril(torch.gather(
            st_par.unsqueeze(-2).expand(-1, -1, self.tril_template.shape[0], -1),
            -1,
            self.tril_template.unsqueeze(0).unsqueeze(0).expand(st_par.shape[0], st_par.shape[1], -1, -1),
        ))
        diag_activated = torch.nn.functional.softplus(torch.diagonal(scale_trils_raw, dim1=-2, dim2=-1).clamp(min=-15))
        scale_trils = torch.diagonal_scatter(scale_trils_raw, diag_activated, dim1=-2, dim2=-1)
        component_dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=loc,
            scale_tril=scale_trils,
        )
        mixture_weight_dist = torch.distributions.Categorical(logits = pi.clamp(min=-15))
        mixture_dist = torch.distributions.mixture_same_family.MixtureSameFamily(mixture_weight_dist, component_dist)
        return mixture_dist

    def log_likelihood(self, p, y, min_log_proba=-np.inf):
        mixture_dist = self.get_dist(p)
        return mixture_dist.log_prob(y).clamp(min=min_log_proba)

class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, n_components):
        super(MDN, self).__init__()
        self.n_components = n_components
        self.output_dim = output_dim
        self.mdn_mv_util = MDNMV(n_components, output_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, n_components * (1 + self.mdn_mv_util.n_params_per_comp))
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    

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

    net = MDN(1, 2, 100).to(device)
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=10)

    X_train_s, y_train_s, scalers = prepare_training(net, train_X, train_y)

    for k in range(100):
        for ixs in tqdm(get_batch_ixs(X_train_s, batch_size, permute=True)):
            X = X_train_s[ixs]
            y = y_train_s[ixs]
            x = net.layers(X)

            loss = -net.mdn_mv_util.log_likelihood(x, y, min_log_proba=-15).sum()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
        plt.figure()
        p = net(torch.tensor([[0.25]]).float().to(device))
        plt.hist2d(*net.mdn_mv_util.get_dist(p).sample((10000,)).squeeze().T.cpu().numpy(), bins=100, cmap='inferno') # type: ignore
        plt.gca().set_aspect(1)
        plt.savefig("results/mdn_100/%d.png"%k)
        plt.show()
        plt.close()

        if k%10==9:
            fig, axs = plt.subplots(4, 4, figsize=(10, 10))
            (Xscaler, yscaler) = scalers
            for i, ax in zip(np.linspace(0, 2, 4*4+1)[1:]**2, axs.reshape(-1)):
                irt = Xscaler.transform([[i]]) # type: ignore
                p = net(torch.tensor(irt).float().to(device))
                sample = yscaler.inverse_transform(net.mdn_mv_util.get_dist(p).sample((100000,)).squeeze().cpu().numpy()) # type: ignore
                ax.hist2d(*sample.T, bins=100, cmap='inferno') # type: ignore
                ax.set_aspect('equal')
                ax.set_title("$f^{-1}(%.2f)$"%i)
                ax.set_xlim([-3, 3])
                ax.set_ylim([-3, 3])
                ax.set_facecolor('k')
            fig.tight_layout()
            plt.savefig("results/mdn_100/%d_big.png"%k)
            plt.show()
            plt.close()
            torch.save(net.state_dict(), "results/mdn_100/%d.pt"%k)