import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import pandas as pd
import argparse
import os
import torch
from tqdm.auto import tqdm
from glob import glob
import lib.utils as utils
import lib.crps as crps
from copy import deepcopy
import torch.nn as nn
import time

start_time = time.time()

parser = argparse.ArgumentParser(description="Process some flags.")

parser.add_argument('--expname', type=str, required=True, help='Name of the experiment')
parser.add_argument('--device', type=str, default='auto', help='Device to use (default: auto)')

args = parser.parse_args()

if args.device == 'auto':
    device="cuda:" + str(utils.select_gpu_with_low_usage())
else:
    device=args.device
print("using device", device)

results_dir = "results/"+args.expname
os.makedirs(results_dir, exist_ok=True)

default_params = {
    'l2reg': 1e-6,
    'splits': 20,
}

dataset_param_overrides = {
    'protein-tertiary-structure': {
        'splits': 5,
    },
    'concrete': {
        'l2reg': 1e-4,
    },
    'bostonHousing': {
        'l2reg': 1e-4,
    },
    'energy': {
        'l2reg': 1e-4,
    },
    'yacht': {
        'l2reg': 1e-4,
    },
}

class CRPSRegressor(nn.Module):
    def __init__(self, input_dim, nheads):
        super(CRPSRegressor, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 50),
            nn.LayerNorm(50),
            nn.GELU(),
            crps.EpsilonSampler(100),
            nn.Linear(50+100, 50),
            nn.LayerNorm(50),
            nn.GELU(),
        ])
        self.samples = nn.Linear(50, nheads)
        self.weights = nn.Linear(50, nheads)

    def forward(self, x, n_samples=10):
        for l in self.layers:
            if type(l)==crps.EpsilonSampler:
                x = l(x, n_samples=n_samples)
            else:
                x = l(x)
        samples = self.samples(x)
        weights_raw = self.weights(x)
        weights_log_softmax = nn.functional.log_softmax(weights_raw.reshape(*weights_raw.shape[:-2], -1), dim=-1)
        weights = torch.exp(weights_log_softmax+np.log(weights_log_softmax.shape[-1]))
        return samples.reshape(*samples.shape[:-2], -1), weights

class Ensemble(nn.Module):
    def __init__(self, n_networks, *args, **kwargs):
        super(Ensemble, self).__init__()
        self.networks = nn.ModuleList([
            CRPSRegressor(*args, **kwargs) for i in range(n_networks)
        ])

    def forward(self, *args, **kwargs):
        preds, weights = zip(*[network(*args, **kwargs) for network in self.networks])
        return torch.cat([p.unsqueeze(-2) for p in preds], dim=-2), torch.cat([w.unsqueeze(-2) for w in weights], dim=-2)

batch_size=16
optimizer=None
samples=100
max_epoch=1000
valid_samples=1000
l2reg=1e-6
verbose=True
max_patience=50
heads=20
n_networks=5


data_repo_dir = "./DropoutUncertaintyExps/UCI_Datasets"
dataset_names = [dir.split("/")[-1] for dir in glob(data_repo_dir+"/*")]

all_scores = []
for dataset_name in dataset_names:
    params = dict(default_params, **dataset_param_overrides.get(dataset_name, {}))
    data_dir = data_repo_dir + "/%s/data/"%dataset_name
    data_path = data_dir + "data.txt"

    print("starting", dataset_name)

    #
    # original setup code start ///
    np.random.seed(1)
    data = np.loadtxt(data_path)

    def _get_index_train_test_path(split_num, train = True):
        if train:
            return data_dir + "index_train_" + str(split_num) + ".txt"
        else:
            return data_dir + "index_test_" + str(split_num) + ".txt" 

    splits = []
    for split in range(params['splits']):
        index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
        index_test = np.loadtxt(_get_index_train_test_path(split, train=False))
        splits.append((index_train, index_test))

    ix_features = np.loadtxt(data_dir + "index_features.txt")
    ix_target = np.loadtxt(data_dir + "index_target.txt")

    X = data[ : , [int(i) for i in ix_features.tolist()] ]
    y = data[ : , int(ix_target.tolist()) ]
    # /// original setup code end
    #

    rmses=[]
    nlls=[]
    crpss=[]
    for split_ix, (index_train, index_test) in enumerate(tqdm(splits)):
        X_train = X[ [int(i) for i in index_train.tolist()] ]
        y_train = y[ [int(i) for i in index_train.tolist()] ]
        X_test = X[ [int(i) for i in index_test.tolist()] ]
        y_test = y[ [int(i) for i in index_test.tolist()] ]

        X_train_original = X_train
        y_train_original = y_train
        num_training_examples = int(0.8 * X_train.shape[0])
        X_validation = X_train[num_training_examples:, :]
        y_validation = y_train[num_training_examples:]
        X_train = X_train[0:num_training_examples, :]
        y_train = y_train[0:num_training_examples]

        batch_size=int(np.sqrt(data.shape[0])-5)
        l2reg=params['l2reg']

        model = Ensemble(n_networks, X_train.shape[1], heads).to(device)
        X_train_s, y_train_s, scalers = crps.prepare_training(model, X_train, y_train)

        optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=l2reg) if optimizer is None else optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=10)

        patience = 0
        best_val_loss = np.inf
        best_val_loss2 = np.inf
        best_model = None

        for k in range(max_epoch):
            losses = []
            model.train()

            for ixs in utils.get_batch_ixs(X_train_s, batch_size, permute=True):
                preds, weights = model(X_train_s[ixs], n_samples=samples)
                loss = crps.crps_loss_weighted(preds, weights, y_train_s[ixs].unsqueeze(-1)).sum()

                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
            scheduler.step()

            model.eval()
            val_pred_samples, val_pred_weights = crps.predict(model, scalers, X_validation, n_samples=valid_samples//(heads*n_networks), batch_size=128)
            with torch.device(X_train_s.device):
                preds_gpu, weights_gpu, y_gpu = torch.tensor(val_pred_samples).float(), torch.tensor(val_pred_weights).float(), torch.tensor(y_validation.reshape(-1,1)).float()
                val_crps = crps.crps_loss_weighted(preds_gpu, weights_gpu, y_gpu).mean().item()
                val_nll = crps.nll_gpu_weighted(preds_gpu, weights_gpu, y_gpu).mean().item()
                val_expectation = (preds_gpu*weights_gpu).mean(dim=-1)
                val_rmse = ((y_gpu.squeeze()-val_expectation)**2).mean().item()**(1/2)

            print(k, val_crps, val_rmse, val_nll) if verbose else None

            if(best_val_loss > val_nll):
                best_val_loss = val_nll
                patience = 0
                best_model = deepcopy(model.state_dict())
            elif(best_val_loss2 > val_crps):
                best_val_loss2 = val_crps
                patience = 0
            else:
                patience += 1
            if(patience > max_patience):
                print("breaking at", k)
                break
        assert best_model is not None  
        model.load_state_dict(best_model)

        
        val_pred_samples, val_pred_weights = crps.predict(model, scalers, X_validation, n_samples=valid_samples//heads, batch_size=128)
        with torch.device(X_train_s.device):
            preds_gpu, weights_gpu, y_gpu = torch.tensor(val_pred_samples).float(), torch.tensor(val_pred_weights).float(), torch.tensor(y_validation.reshape(-1,1)).float()
            val_crps = crps.crps_loss_weighted(preds_gpu, weights_gpu, y_gpu).mean().item()
            val_nll = crps.nll_gpu_weighted(preds_gpu, weights_gpu, y_gpu, batch_size=4).mean().item()
            val_expectation = (preds_gpu*weights_gpu).mean(dim=-1)
            val_rmse = ((y_gpu.squeeze()-val_expectation)**2).mean().item()**(1/2)
        print("valid", val_crps, val_rmse, val_nll) if verbose else None

        test_pred_samples, test_pred_weights = crps.predict(model, scalers, X_test, n_samples=valid_samples//heads, batch_size=128)
        with torch.device(X_train_s.device):
            preds_gpu, weights_gpu, y_gpu = torch.tensor(test_pred_samples).float(), torch.tensor(test_pred_weights).float(), torch.tensor(y_test.reshape(-1,1)).float()
            test_crps = crps.crps_loss_weighted(preds_gpu, weights_gpu, y_gpu).mean().item()
            test_nll = crps.nll_gpu_weighted(preds_gpu, weights_gpu, y_gpu, batch_size=4).mean().item()
            test_expectation = (preds_gpu*weights_gpu).mean(dim=-1)
            test_rmse = ((y_gpu.squeeze()-test_expectation)**2).mean().item()**(1/2)
        print('test', test_crps, test_rmse, test_nll) if verbose else None

        torch.save(model.state_dict(), results_dir+"/%s_%d.pt"%(dataset_name, split_ix))
        all_scores.append((dataset_name, split_ix, test_crps, test_rmse, test_nll))
        rmses.append(test_rmse)
        nlls.append(test_nll)
        crpss.append(test_crps)
    
    print(dataset_name, np.mean(crpss), np.mean(rmses), np.mean(nlls))
    pd.DataFrame(all_scores, columns=['dataset', 'split', 'crps', 'rmse', 'nll']).to_csv(results_dir+'/scores.csv', index=False)

end_time = time.time()
running_time = end_time - start_time
with open(results_dir+"/time.txt", "w") as file:
    file.write(str(running_time))
