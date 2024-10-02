import torch
from torch import nn
import numpy as np
import lib.utils as utils
import lib.mdn as mdn
from copy import deepcopy

def predict(net, scalers, X, batch_size=32):
    device = next(net.parameters()).device
    (Xscaler, yscaler) = scalers
    X_s = torch.tensor(Xscaler.transform(X), device=device, dtype=torch.float)
    preds = []
    with torch.no_grad():
        for ixs in utils.get_batch_ixs(X_s, batch_size):
            pred = net(X_s[ixs])
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return np.concatenate([
        yscaler.inverse_transform(preds[:, :, 0])[:, :, None], # mu is fully reverse transformed
        preds[:, :, 1][:, :, None] * yscaler.scale_, # sigma only needs to be scaled
    ], axis=2)

def eval(net, scalers, X_eval, y_eval):
    eval_pred = predict(net, scalers, X_eval)
    eval_expectation = eval_pred[:, :, 0].mean(axis=1)
    eval_variance = ((eval_pred**2).sum(axis=2).mean(axis=1)-eval_expectation**2)**(1/2)
    eval_rmse = ((eval_expectation-y_eval)**2).mean()**(1/2)
    eval_nll = -torch.distributions.Normal(torch.tensor(eval_expectation), torch.tensor(eval_variance)).log_prob(torch.tensor(y_eval)).mean()
    return eval_rmse.item(), eval_nll.item()

def train(
        net,
        X_train,
        y_train,
        X_validation,
        y_validation,
        batch_size=16,
        optimizer=None,
        max_epoch=1000,
        max_patience=20,
        l2reg=1e-9,
        verbose=False,
        lr=0.001,
    ):

    device = next(net.parameters()).device
    X_train_s, y_train_s, scalers = mdn.prepare_training(net, X_train, y_train)

    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2reg) if optimizer is None else optimizer
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=10)

    patience = 0
    best_val_loss = np.inf
    best_model_params = None
    for k in range(max_epoch):
        losses = []
        net.train()
        for ixs in utils.get_batch_ixs(X_train_s, batch_size, permute=True):
            preds = net(X_train_s[ixs])
            loss = -torch.distributions.Normal(preds[:, :, 0], preds[:, :, 1]).log_prob(y_train_s[ixs].expand(-1, preds.shape[1])).sum()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            optim.step()
            losses.append(loss.item())
        scheduler.step()
        net.eval()

        val_rmse, val_nll = eval(net, scalers, X_validation, y_validation)
        if verbose:
            print(k, np.mean(losses), val_nll, val_rmse)

        if(best_val_loss > val_nll):
            best_val_loss = val_nll
            patience = 0
            best_model_params = deepcopy(net.state_dict())
        else:
            patience += 1
        if(patience > max_patience):
            if verbose:
                print("breaking at", k)
            break

    net.load_state_dict(best_model_params)
    return net, scalers
