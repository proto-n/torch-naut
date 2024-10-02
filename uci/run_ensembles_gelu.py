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
import lib.ensembles as ensembles
import torch
import torch.nn as nn
from lib import mdn
import time

start_time = time.time()

class PNN(nn.Module):
    def __init__(self, input_dim):
        super(PNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.LayerNorm(50),
            nn.GELU(),
            nn.Linear(50, 50),
            nn.LayerNorm(50),
            nn.GELU(),
            nn.Linear(50, 3)
        )

    def forward(self, x):
        x = self.layers(x).view(x.shape[0], 1, 3)
        return mdn.transform_output(x)
    
class Ensemble(nn.Module):
    def __init__(self, input_dim, n_networks):
        super(Ensemble, self).__init__()
        self.networks = nn.ModuleList([
            PNN(input_dim) for i in range(n_networks)
        ])

    def forward(self, x):
        return torch.cat([
            network(x)[:, :, :2] for network in self.networks
        ], dim=1)
    
parser = argparse.ArgumentParser(description="Process some flags.")

parser.add_argument('--expname', type=str, required=True, help='Name of the experiment')
parser.add_argument('--n_networks', type=int, required=True, help='Number of networks')
parser.add_argument('--min_std', type=float, default=0.0, help='Minimum standard deviation (default: 0.0)')
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
 
        net = Ensemble(X_train.shape[1], args.n_networks).to(device)
        net, scalers = ensembles.train(
            net,
            X_train,
            y_train,
            X_validation,
            y_validation,
            batch_size=int(np.sqrt(data.shape[0])-5),
            l2reg=params['l2reg'],
            max_patience=50,
            verbose=True,
        )

        val_rmse, val_nll = ensembles.eval(net, scalers, X_validation, y_validation)
        print("val", val_rmse, val_nll)

        test_rmse, test_nll = ensembles.eval(net, scalers, X_test, y_test)
        print("test", test_rmse, test_nll)

        all_scores.append((dataset_name, split_ix, test_rmse, test_nll))
        torch.save(net.state_dict(), results_dir+"/%s_%d.pt"%(dataset_name, split_ix))

pd.DataFrame(all_scores, columns=['dataset', 'split', 'rmse', 'nll']).to_csv(results_dir+'/scores.csv', index=False)

end_time = time.time()
running_time = end_time - start_time
with open(results_dir+"/time.txt", "w") as file:
    file.write(str(running_time))
