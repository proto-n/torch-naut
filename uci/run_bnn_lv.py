import sys
import os
sys.path.append('lightning-uq-box')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import lib.crps as crps
import lib.utils as utils


import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from functools import partial
import argparse

from uq_method_box.models import MLP
from uq_method_box.uq_methods import BNN_LV_VI_Batched

from tqdm.auto import tqdm
from glob import glob
import time


start_time = time.time()



torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(description="Process some flags.")

parser.add_argument('--expname', type=str, required=True, help='Name of the experiment')
parser.add_argument('--device', type=str, default='auto', help='Device to use (default: auto)')
parser.add_argument('--max_epoch', type=int, default=1000, help='Max epochs to run')

args = parser.parse_args()

if args.device == 'auto':
    device="cuda:" + str(utils.select_gpu_with_low_usage())
else:
    device=args.device
print("using device", device)

results_dir = "results/"+args.expname
os.makedirs(results_dir, exist_ok=True)


verbose=True
max_patience=20
max_epoch=args.max_epoch

my_dir = tempfile.mkdtemp()

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

        batch_size=int(np.sqrt(data.shape[0])-5)
        l2reg=params['l2reg']

        my_config = {
            "model_args": {
                "n_inputs": X_train.shape[1],
                "n_outputs": 1,
                "n_hidden": [50, 50],
                "activation_fn": torch.nn.ReLU(),
            },
            "loss_fn": "nll",
            "latent_net": {
                "n_inputs": X_train.shape[1] + 1,  # num_input_features + num_target_dim
                "n_outputs": 2,  # 2 * lv_latent_dimx
                "n_hidden": [20, 20],
                "activation_fn": torch.nn.ReLU(),
            },
        }

        my_dir = tempfile.mkdtemp()

        max_epochs = 2000

        base_model = BNN_LV_VI_Batched(
            model=MLP(**my_config["model_args"]),
            latent_net=MLP(**my_config["latent_net"]),
            optimizer=partial(torch.optim.Adam, lr=1e-2),
            save_dir=my_dir,
            num_training_points=X_train.shape[0],
            part_stoch_module_names=["model.6"],
            latent_variable_intro="first",
            n_mc_samples_train=50,
            n_mc_samples_test=50,
            output_noise_scale=1.3,
            #output_noise_scale=10.3,
            prior_mu=0.0,
            prior_sigma=1.0,
            posterior_mu_init=0.0,
            posterior_rho_init=-2.2522,
            #posterior_rho_init=-6.0,
            alpha=1.0,
            #alpha=1e-03,
        )
        base_model.to(device)


        model_param_list = base_model.exclude_from_wt_decay(base_model.named_parameters(), weight_decay=l2reg)

        optim = torch.optim.AdamW(model_param_list, lr=0.05, weight_decay=params['l2reg'])
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=10)

        X_train_s, y_train_s, (Xscaler, yscaler) = crps.prepare_training(base_model, X_train, y_train)
        X_val_s = torch.tensor(Xscaler.transform(X_validation), device=device, dtype=torch.float)
        y_val_s = torch.tensor(yscaler.transform(y_validation.reshape(-1, 1)), device=device, dtype=torch.float)


        for k in range(max_epoch):
            base_model.train()
            base_model.to(device)
            losses = []
            for ixs in utils.get_batch_ixs(X_train_s, batch_size):
                with torch.device(device):
                    loss = base_model.training_step((X_train_s[ixs], y_train_s[ixs]))
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())

            patience = 0
            best_val_loss = np.inf
            best_model = None
            with torch.no_grad():
                losses = []
                for ixs in utils.get_batch_ixs(X_val_s, batch_size):
                    with torch.device(device):
                        loss = base_model.training_step((X_val_s[ixs], y_val_s[ixs]))
                    losses.append(loss.item())
                val_loss = np.mean(losses)
                print(k, "valid", val_loss) if verbose else None

            if(best_val_loss > val_loss):
                best_val_loss = val_loss
                patience = 0
                best_model = deepcopy(base_model.state_dict())
            else:
                patience += 1
            if(patience > max_patience):
                print("breaking at", k)
                break

        assert best_model is not None  
        base_model.load_state_dict(best_model)

        base_model.cpu()
        base_model.eval()

        X_val_s = X_val_s.cpu()

        preds = []
        for ixs in utils.get_batch_ixs(X_val_s, batch_size):
            preds.append(base_model.predict_step(X_val_s[ixs], n_samples_pred=200)['samples'].squeeze().T)
        val_pred_samples = yscaler.inverse_transform(np.concatenate(preds, axis=0))

        with torch.no_grad():
            with torch.device(X_train_s.device):
                preds_gpu, y_gpu = torch.tensor(val_pred_samples).float(), torch.tensor(y_validation.reshape(-1,1)).float()
                val_crps = crps.crps_loss(preds_gpu, y_gpu).mean().item()
                val_nll = crps.nll_gpu(preds_gpu, y_gpu).mean().item()
                val_expectation = (preds_gpu).mean(dim=-1)
                val_rmse = ((y_gpu.squeeze()-val_expectation)**2).mean().item()**(1/2)
            print("valid", val_crps, val_rmse, val_nll) if verbose else None

        X_test_s = torch.tensor(Xscaler.transform(X_test), dtype=torch.float)

        preds = []
        for ixs in utils.get_batch_ixs(X_test_s, batch_size):
            preds.append(base_model.predict_step(X_test_s[ixs], n_samples_pred=200)['samples'].squeeze().T)
        test_pred_samples = yscaler.inverse_transform(np.concatenate(preds, axis=0))

        with torch.no_grad():
            with torch.device(X_train_s.device):
                preds_gpu, y_gpu = torch.tensor(test_pred_samples).float(), torch.tensor(y_test.reshape(-1,1)).float()
                test_crps = crps.crps_loss(preds_gpu, y_gpu).mean().item()
                test_nll = crps.nll_gpu(preds_gpu, y_gpu).mean().item()
                test_expectation = (preds_gpu).mean(dim=-1)
                test_rmse = ((y_gpu.squeeze()-test_expectation)**2).mean().item()**(1/2)
            print("test", val_crps, val_rmse, val_nll) if verbose else None

        torch.save(base_model.state_dict(), results_dir+"/%s_%d.pt"%(dataset_name, split_ix))
        all_scores.append((dataset_name, split_ix, test_crps, test_rmse, test_nll))
        rmses.append(test_rmse)
        nlls.append(test_nll)
    
        print(dataset_name, np.mean(rmses), np.mean(nlls))
        pd.DataFrame(all_scores, columns=['dataset', 'split', 'test_crps', 'rmse', 'nll']).to_csv(results_dir+'/scores.csv', index=False)

end_time = time.time()
running_time = end_time - start_time
with open(results_dir+"/time.txt", "w") as file:
    file.write(str(running_time))
