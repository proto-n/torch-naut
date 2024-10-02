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



class MDN(nn.Module):
    def __init__(self, input_dim, n_components):
        super(MDN, self).__init__()
        self.n = n_components
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 3*n_components)
        ])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = x.squeeze().view(x.shape[0], self.n, 3)
        # return [batch x mixtures x [mu, sigma, mixture_coef]]
        return torch.cat([
            x[:, :, 0].unsqueeze(2),
            nn.functional.softplus(torch.clamp(x[:, :, 1].unsqueeze(2), min=-15)), # avoiding nans when variance converges to -inf
            torch.clamp(x[:, :, 2].unsqueeze(2), min=-15, max=15) # avoiding nans when logit converges to +-inf
        ], dim=2)

    def log_likelihood(self, p, y, min_log_proba=-15):
        dist = torch.distributions.Normal(p[:, :, 0], p[:, :, 1])
        log_likelihood_terms = dist.log_prob(y.reshape(-1,1))
        log_mixture_logcoefs = nn.functional.log_softmax(p[:, :, 2], dim=1)
        log_likelihoods = torch.logsumexp(log_likelihood_terms + log_mixture_logcoefs, dim=1)
        return torch.clamp(log_likelihoods, min=min_log_proba) # avoiding nans when log prob is extremely low
    
    def log_likelihood2(self, p, y, min_log_proba=-15):
        component_dist = torch.distributions.Normal(p[:, :, 0], p[:, :, 1])
        mixture_weight_dist = torch.distributions.Categorical(logits = p[:, :, 2])
        mixture = torch.distributions.mixture_same_family.MixtureSameFamily(mixture_weight_dist, component_dist)
        return torch.clamp(mixture.log_prob(y), min=min_log_proba)

    def expected_value(self, p):
        mixture_coefs = nn.functional.softmax(p[:, :, 2], dim=1)
        mu = p[:, :, 0]
        return (mixture_coefs*mu).sum(dim=1)
    
    def sample(self, p, samples=100):
        mixture_rand = torch.distributions.categorical.Categorical(logits = p[:, :, 2])
        mixture_element_ixs = mixture_rand.sample((samples,)).T # type: ignore
        mixture_element_params = torch.gather(p, 1, mixture_element_ixs.unsqueeze(-1).expand(-1, -1, 2))
        mixture_element = torch.distributions.normal.Normal(mixture_element_params[:, :, 0], mixture_element_params[:, :, 1])
        return mixture_element.sample((1,)).squeeze() # type: ignore

if __name__ == "__main__":
    start_time = time.time()

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
            if gpu_memory[gpu_id] > min_memory_free and gpu_utilization < 10:  # Adjust utilization threshold as needed
                min_memory_usage = memory_usage
                min_gpu_id = gpu_id

        return min_gpu_id
        
    selected_gpu_id = select_gpu_with_low_usage()
    device="cuda:"+str(selected_gpu_id)
    epochs = 200
    optimizer = torch.optim.SGD
    optimizer_params = dict(lr=0.0001, momentum=0.8)
    train_samples = 100
    model_save_mod = 20
    image_save_mod = 1
    components = 100
    samples_per_column = 5000

    epochs = 2000
    optimizer_params = dict(lr=0.0001, momentum=0.8)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = Image.open('mona.png')

    image_tensor = transform(image)[0][:500, :500] # type: ignore # it's grayscale

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


    res_folder = "results/mdn"
    os.makedirs(res_folder, exist_ok=True)


    l2 = MDN(1, components).to(device)
    optim = optimizer(l2.parameters(), **optimizer_params) # type: ignore
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=10)


    dist_sample_2d =torch.stack([
        dist_sample,
        dist_sample.shape[0]-1-torch.arange(dist_sample.shape[0]).unsqueeze(1).expand(-1, samples_per_column)
    ]).reshape(2, -1)
    train_data = dist_sample_2d.T.float().to(device)
    train_data = train_data*2/image_tensor.shape[0]-1


    print("running on device", device)
    to_plot_X = -((torch.arange(image_tensor.shape[0], device=device).float()/image_tensor.shape[0])*2-1).reshape(-1, 1)
    for k in range(epochs):
        for b in tqdm(torch.tensor_split(train_data[torch.randperm(train_data.shape[0])], train_data.shape[0]//2000)):
            yb, Xb = b.T
            pred = l2(Xb.view(-1,1))
            loss = -l2.log_likelihood(pred, yb).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()

        if(k % image_save_mod == image_save_mod - 1):
            with torch.no_grad():
                out_samp_unif_ = []
                out_samp_unif_weight_ = []
                for tpb in torch.tensor_split(to_plot_X, to_plot_X.shape[0]//10):
                    samp = l2.sample(l2(tpb), 10000)
                    out_samp_unif_.append(samp)
                t_sample = torch.concatenate(out_samp_unif_, dim=0)
                t_sample = t_sample.view(t_sample.shape[0], -1)
                t_coord_sample = samples_to_coords(t_sample, norm=True)

                weights = image_tensor.sum(dim=1).unsqueeze(1).expand(-1, t_sample.shape[1]).reshape(-1)
                full_weights = (weights/weights.max())

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

    end_time = time.time()
    running_time = end_time - start_time
    with open(res_folder+"/time.txt", "w") as file:
        file.write(str(running_time))
