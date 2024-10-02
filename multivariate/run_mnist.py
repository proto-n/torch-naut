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
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 16),
            crps.EpsilonSampler(4),
            nn.Linear(20, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_dim)
        ])

    def forward(self, x, n_samples=10):
        for l in self.layers:
            if type(l)==crps.EpsilonSampler:
                x = l(x, n_samples=n_samples)
            else:
                x = l(x)
        return nn.functional.leaky_relu(x)
    
if __name__ == "__main__":
    device=utils.select_gpu_with_low_usage()
    print("running on", device)

    train_x = torch.load('mnist_data/train_x.pt')
    train_y = torch.load('mnist_data/train_y.pt')
    test_x = torch.load('mnist_data/test_x.pt')
    test_y = torch.load('mnist_data/test_y.pt')

    train_x = train_x.reshape(train_x.shape[0], -1).to(device)
    train_y = torch.nn.functional.one_hot(train_y.type(torch.int64)).float().to(device)
    test_x = train_x.reshape(test_x.shape[0], -1).to(device)
    test_y = torch.nn.functional.one_hot(test_y.type(torch.int64), num_classes=train_y.shape[1]).float().to(device)

    net = CRPSModel(train_y.shape[1], train_x.shape[1]).to(device)
    optim = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=20)

    def show_images(images, title):
        cols = 5
        rows = int(len(images)/cols) + 1
        plt.figure(figsize=(8,8))
        index = 1
        for x in images:
            plt.subplot(rows, cols, index)
            plt.imshow(x, cmap=plt.cm.gray) # type: ignore
            plt.axis('off')
            index += 1
        plt.suptitle(title, fontsize=15)
        plt.tight_layout()
        plt.show()

    for k in range(1000):
        losses = []
        for ixs in tqdm(utils.get_batch_ixs(train_x, 16, permute=True)):
            t_sample = net(train_y[ixs], n_samples=50)
            loss = crps.crps_loss_mv(t_sample, train_x[ixs]).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        scheduler.step()
        print(k, np.mean(losses), torch.where(train_y[ixs[0]])[0][0].item()) # type: ignore
        if k%20==19:
            show_images(t_sample[0][:50].reshape(-1, 28, 28).detach().cpu().numpy(), torch.where(train_y[ixs[0]])[0][0].item()) # type: ignore
            plt.savefig('results/mnist/%d.png'%k)
            plt.close()
        if k%100==99:
            torch.save(net.state_dict(), 'results/mnist/%d.pt'%k)
