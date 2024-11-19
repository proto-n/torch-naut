# %%
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
import lib.crps as crps
from copy import deepcopy
import torch.nn as nn
import time

start_time = time.time()

device="cuda:0"

print("using device", device)

results_dir = "results/scaling"
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

class MDN(nn.Module):
    def __init__(self, input_dim, n_components, n_layers, n_layer_size, min_std=0):
        super(MDN, self).__init__()
        self.n_components = n_components
        self.min_std = min_std
        self.layers = nn.Sequential(*([
            nn.Linear(input_dim, n_layer_size),
            nn.LayerNorm(n_layer_size),
            nn.GELU(),
        ] + [
            nn.Sequential(
                nn.Linear(n_layer_size, n_layer_size),
                nn.LayerNorm(n_layer_size),
                nn.GELU(),
            ) for _ in range(n_layers)
        ] + [
            nn.Linear(n_layer_size, 50),
            nn.LayerNorm(50),
            nn.GELU(),
            nn.Linear(50, 3 * n_components)
        ]))

    def forward(self, x):
        x = self.layers(x).view(x.shape[0], self.n_components, 3)
        return mdn.transform_output(x, min_std=self.min_std)

class CRPSRegressor(nn.Module):
    def __init__(self, input_dim, nheads, n_preprocess_layers, n_preprocess_layer_size):
        super(CRPSRegressor, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, n_preprocess_layer_size),
            nn.LayerNorm(n_preprocess_layer_size),
            nn.GELU(),
        ] + [
            nn.Sequential(
                nn.Linear(n_preprocess_layer_size, n_preprocess_layer_size),
                nn.LayerNorm(n_preprocess_layer_size),
                nn.GELU(),
            ) for _ in range(n_preprocess_layers)
        ] + [
            crps.EpsilonSampler(100),
            nn.Linear(n_preprocess_layer_size+100, 100),
            nn.LayerNorm(100),
            nn.GELU(),
            nn.Linear(100, 50),
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

batch_size=16
optimizer=None
samples=100
max_epoch=1000
valid_samples=1000
l2reg=1e-6
verbose=True
max_patience=50
heads=1


data_repo_dir = "./DropoutUncertaintyExps/UCI_Datasets"
dataset_names = [dir.split("/")[-1] for dir in glob(data_repo_dir+"/*")]

dataset_name = "protein-tertiary-structure"

# %%
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

# %%
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

    model = CRPSRegressor(X_train.shape[1], heads, 1, 50).to(device)
    X_train_s, y_train_s, scalers = crps.prepare_training(model, X_train, y_train)
    break

# %%
sizes = [
    (1, 50),
    (2, 75),
    (3, 100),
    (4, 150),
    (5, 200),
    (6, 250),
    (7, 300),
    (8, 400),
    (9, 500),
    (10, 750),
    (11, 1000),
    (12, 1500),
    (13, 2000),
    (14, 3000),
]

# %%
crps_data = []
for numl, sizel in sizes:
    model = CRPSRegressor(X_train.shape[1], heads, numl, sizel).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=l2reg)
    start_time = time.time()
    for i in range(10):
        for ixs in utils.get_batch_ixs(X_train_s, batch_size, permute=True):
            preds, weights = model(X_train_s[ixs], n_samples=samples)
            loss = crps.crps_loss_weighted(preds, weights, y_train_s[ixs]).sum()
            optim.zero_grad()
            loss.backward()
            optim.step()
    end_time = time.time()
    running_time = end_time - start_time
    print(running_time, sum(p.numel() for p in model.parameters()))
    crps_data.append((sum(p.numel() for p in model.parameters()), running_time))

# %%
pd.DataFrame(crps_data, columns=['nparams', 'time']).to_csv(results_dir+'/crps.csv')

# %%
mdn_data = []
for numl, sizel in sizes:
    net = MDN(X_train.shape[1], 100, numl, sizel).to(device)
    net.train()
    optim = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=l2reg)
    start_time = time.time()
    for i in range(10):
        for ixs in utils.get_batch_ixs(X_train_s, batch_size, permute=True):
            preds = net(X_train_s[ixs])
            loss = -mdn.log_likelihood(preds, y_train_s[ixs], min_log_proba=-20).sum()

            optim.zero_grad()
            loss.backward()
            optim.step()
    end_time = time.time()
    running_time = end_time - start_time
    print(running_time, sum(p.numel() for p in net.parameters()))
    mdn_data.append((sum(p.numel() for p in net.parameters()), running_time))

# %%
pd.DataFrame(mdn_data, columns=['nparams', 'time']).to_csv(results_dir+'/mdn.csv')

# %%



