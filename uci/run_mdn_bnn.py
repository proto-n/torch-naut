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
import lib.mdn as mdn
import torch.nn as nn
from lib import mdn
from lib.bnn import BayesianLayer
import time

start_time = time.time()

class MDN(nn.Module):
    def __init__(self, input_dim, n_components, min_std=0):
        super(MDN, self).__init__()
        self.n_components = n_components
        self.min_std = min_std
        self.layers = nn.Sequential(
            BayesianLayer(input_dim, 50),
            nn.ReLU(),
            BayesianLayer(50, 50),
            nn.ReLU(),
            BayesianLayer(50, 3 * n_components)
        )

    def forward(self, x):
        x = self.layers(x).view(x.shape[0], self.n_components, 3)
        return mdn.transform_output(x, min_std=self.min_std)

parser = argparse.ArgumentParser(description="Process some flags.")

parser.add_argument('--expname', type=str, required=True, help='Name of the experiment')
parser.add_argument('--n_components', type=int, required=True, help='Number of components')
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

        net = MDN(X_train.shape[1], args.n_components, args.min_std).to(device)
        net, scalers = mdn.bnn_train(
            net,
            X_train,
            y_train,
            X_validation,
            y_validation,
            batch_size=int(np.sqrt(data.shape[0])-5),
            l2reg=params['l2reg'],
            max_patience=50,
            kl_coef=0, # much better scores than with kl
            verbose=True,
        )

        val_rmse, val_nll = mdn.bnn_eval(net, scalers, X_validation, y_validation, 100)
        print("val", val_rmse, val_nll)
        test_rmse, test_nll = mdn.bnn_eval(net, scalers, X_test, y_test, 100)
        print("test", test_rmse, test_nll)

        all_scores.append((dataset_name, split_ix, test_rmse, test_nll))
        torch.save(net.state_dict(), results_dir+"/%s_%d.pt"%(dataset_name, split_ix))

pd.DataFrame(all_scores, columns=['dataset', 'split', 'rmse', 'nll']).to_csv(results_dir+'/scores.csv', index=False)

end_time = time.time()
running_time = end_time - start_time
with open(results_dir+"/time.txt", "w") as file:
    file.write(str(running_time))
