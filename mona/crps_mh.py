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
from torchvision import transforms
from PIL import Image
import sys
import subprocess
import os
import time
import lib.crps as crps


class AULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(AULayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, n_samples=10):
        eps = torch.randn(x.shape[0], n_samples, self.out_features-self.in_features, device=x.device)
        return torch.concatenate([x.unsqueeze(1).expand(-1, n_samples, -1), eps], dim=2)


class CRPSRegressor(nn.Module):
    def __init__(self, input_dim, nheads, nens):
        super(CRPSRegressor, self).__init__()
        self.nheads = nheads
        self.nens = nens
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            AULayer(512, 512+20),
            nn.Linear(512+20, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
        ])
        self.samples = nn.Linear(1024, nheads*nens)
        self.weights = nn.Linear(1024, nheads*nens)

    def forward(self, x, n_samples=10):
        for l in self.layers:
            if type(l)==AULayer:
                x = l(x, n_samples=n_samples)
            else:
                x = l(x)
        samples = self.samples(x).transpose(-1, -2).reshape(x.shape[0], self.nens, -1)
        weights_raw = self.weights(x).transpose(-1, -2).reshape(x.shape[0], self.nens, -1)

        weights_log_softmax = nn.functional.log_softmax(weights_raw, dim=-1)
        weights = torch.exp(weights_log_softmax+np.log(weights_log_softmax.shape[-1]))
        return samples, weights
        

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open('mona.png')

    image_tensor = transform(image)[0][:500, :500] # type: ignore

    samples_per_column = 5000
    image_dist = torch.distributions.Categorical(image_tensor/image_tensor.sum(dim=1, keepdim=True))
    dist_sample = image_dist.sample((samples_per_column,)).T # type: ignore

    def samples_to_coords(samples, norm=False):
        ax1c = samples.shape[0]-1-torch.arange(samples.shape[0]).unsqueeze(1).expand(-1, samples.shape[1])
        if norm:
            ax1c = ax1c*2/dist_sample.shape[0]-1
        return torch.stack([
            samples,
            ax1c.to(samples.device),
        ]).reshape(2, -1)

    coord_samples = samples_to_coords(dist_sample)
    weights = image_tensor.sum(dim=1).unsqueeze(1).expand(-1, dist_sample.shape[1]).reshape(-1)


    plt.hist2d(*samples_to_coords(dist_sample).float().numpy(), weights=weights, bins=image_tensor.shape, cmap='gray');
    plt.gca().set_aspect(1)
    plt.axis("off")
    plt.show()

    def crps_loss(yps, y):
        ml=yps.shape[1]
        mrank = torch.argsort(torch.argsort(yps, dim=1))
        return ((2/(ml*(ml-1)))*(yps-y)*(((ml-1)*(y<yps))-mrank)).sum(axis=1)

    epochs = 2000

    def select_gpu_with_low_usage():
        device_ids = list(range(torch.cuda.device_count()))
        min_memory_free = 20000
        min_gpu_id = -1

        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
        gpu_memory = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]

        for gpu_id in device_ids:
            torch.cuda.set_device(gpu_id)
            memory_usage = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            gpu_utilization = torch.cuda.utilization(gpu_id)
            if gpu_memory[gpu_id] > min_memory_free and gpu_utilization < 10:
                min_memory_usage = memory_usage
                min_gpu_id = gpu_id

        return min_gpu_id

    selected_gpu_id = select_gpu_with_low_usage()
    device="cuda:"+str(selected_gpu_id)

    res_folder = "results/crps_mh"
    os.makedirs(res_folder, exist_ok=True)

    epochs = 200
    optimizer = torch.optim.SGD
    optimizer_params = dict(lr=0.0001, momentum=0.8)
    model_save_mod = 20
    image_save_mod = 1

    heads = 50
    nens = 1
    train_samples = 20

    load = None

    l2 = CRPSRegressor(1, heads, nens).to(device)
    optim = optimizer(l2.parameters(), **optimizer_params) # type: ignore
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=10)

    train_data = coord_samples.T.float().to(device)
    train_data = train_data*2/image_tensor.shape[0]-1

    epochs = 2000

    print("running on device", device)
    to_plot_X = -((torch.arange(image_tensor.shape[0], device=device).float()/image_tensor.shape[0])*2-1).reshape(-1, 1)
    for k in range(epochs):
        print(k)
        if load is not None:
            if k<load:
                scheduler.step()
                continue
            l2.load_state_dict(torch.load(res_folder+"/%d.pt"%load))
            load = None

        for b in tqdm(torch.tensor_split(train_data[torch.randperm(train_data.shape[0])], train_data.shape[0]//2000)):
            yb, Xb = b.T
            t_sample,  t_weight = l2(Xb.view(-1,1), n_samples=train_samples)
            loss = crps.crps_loss_weighted(t_sample, t_weight, yb.reshape(-1,1,1)).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()

        if(k % image_save_mod == image_save_mod - 1):
            with torch.no_grad():
                out_samp_unif_ = []
                out_samp_unif_weight_ = []
                for tpb in torch.tensor_split(to_plot_X, to_plot_X.shape[0]//10):
                    samp, weight = l2(tpb, n_samples=10000//(heads*nens))
                    out_samp_unif_.append(samp)
                    out_samp_unif_weight_.append(weight)
                t_sample = torch.concatenate(out_samp_unif_, dim=0)
                t_weight = torch.concatenate(out_samp_unif_weight_, dim=0)
                t_sample = t_sample.view(t_sample.shape[0], -1)
                t_weight = t_weight.view(t_weight.shape[0], -1)
                t_coord_sample = samples_to_coords(t_sample, norm=True)

                weights = image_tensor.sum(dim=1).unsqueeze(1).expand(-1, t_sample.shape[1]).reshape(-1)
                full_weights = (weights/weights.max())*t_weight.reshape(-1).cpu()
                plt.figure()
                plt.hist2d(*t_coord_sample.cpu().numpy(), weights=full_weights, bins=(np.linspace(-1.1, 1.1, int(image_tensor.shape[0]*1.2)), np.linspace(-1, 1, image_tensor.shape[1])), cmap='gray');
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.gca().set_aspect(1)
                plt.axis("off")
                plt.savefig(res_folder+"/%d.png"%k)
                plt.show()
                plt.close()

        if(k%model_save_mod==model_save_mod-1):
            torch.save(l2.state_dict(), res_folder+"/%d.pt"%k)