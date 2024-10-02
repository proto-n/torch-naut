import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from . import utils
from . import bnn

# expects an x tensor of [batch x num_components x 3] with the last
# three dims corresponding to (mu, s, pi) and calculates
# activation functions / clamping
def transform_output(x, min_std=0):
    return torch.cat([
        x[:, :, 0].unsqueeze(2), # mu
        nn.functional.softplus(torch.clamp(x[:, :, 1].unsqueeze(2), min=-15)) + min_std,
        torch.clamp(x[:, :, 2].unsqueeze(2), min=-15, max=15)
    ], dim=2)

# expects p in the form returned by transform_output(), y targets
# and returns log likelihood with optional clamping to be activated
# when used as a loss
def log_likelihood(p, y, min_log_proba=-np.inf):
    dist = torch.distributions.Normal(p[:, :, 0], p[:, :, 1])
    log_likelihood_terms = dist.log_prob(y.reshape(-1,1))
    mixture_logcoefs = nn.functional.log_softmax(p[:, :, 2], dim=1)
    mixture_logterms = log_likelihood_terms + mixture_logcoefs # [log(w * likelihood)]
    log_likelihoods = torch.logsumexp(mixture_logterms, dim=1)
    return torch.clamp(log_likelihoods, min=min_log_proba) # avoiding nans when log prob is extremely low

# expects p in the form returned by transform_output() and calculates
# the expected value of the mixture
def expected_value(p):
    mixture_coefs = nn.functional.softmax(p[:, :, 2], dim=1)
    mu = p[:, :, 0]
    return (mixture_coefs*mu).sum(dim=1)


# expects p in the form returned by transform_output() and returns
# n samples from the distributions
def sample(p, n=100):
    mixture_rand = torch.distributions.categorical.Categorical(logits = p[:, :, 2])
    mixture_element_ixs = mixture_rand.sample((n,)).T
    mixture_element_params = torch.gather(p, 1, mixture_element_ixs.unsqueeze(-1).expand(-1, -1, 2))
    mixture_element = torch.distributions.normal.Normal(mixture_element_params[:, :, 0], mixture_element_params[:, :, 1])
    return mixture_element.sample((1,)).squeeze()

# expects torch nn, numpy training sets, returns scaled training tensors
# and scalers
def prepare_training(net, X_train, y_train):
    device = next(net.parameters()).device
    Xscaler = StandardScaler()
    yscaler = StandardScaler()
    X_train_s = torch.tensor(Xscaler.fit_transform(X_train), device=device, dtype=torch.float)
    y_train_s = torch.tensor(yscaler.fit_transform(y_train.reshape(-1,1)), device=device, dtype=torch.float)
    return X_train_s, y_train_s, (Xscaler, yscaler)

# expects torch nn, scalers returned by prepare_training() and X np array
# returns scaled y preds compatible with return format of transform_output()
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
        preds[:, :, 2][:, :, None], # mixture weights need no scaling
    ], axis=2)

def eval(net, scalers, X_eval, y_eval):
    net.eval()
    device = next(net.parameters()).device

    eval_pred = predict(net, scalers, X_eval)
    eval_nll = -log_likelihood(
        torch.tensor(eval_pred, device=device),
        torch.tensor(y_eval, device=device),
    ).mean()
    eval_expectation = expected_value(torch.tensor(eval_pred, device=device)).cpu().numpy()
    eval_rmse = ((eval_expectation-y_eval)**2).mean()**(1/2)

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
    X_train_s, y_train_s, scalers = prepare_training(net, X_train, y_train)

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
            loss = -log_likelihood(preds, y_train_s[ixs], min_log_proba=-20).sum()

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


def bnn_eval(net, scalers, X_eval, y_eval, num_evals):
    net.eval()
    device = next(net.parameters()).device

    eval_preds = [predict(net, scalers, X_eval) for i in range(num_evals)]
    eval_nll = -(torch.logsumexp(torch.stack([log_likelihood(
        torch.tensor(vp, device=device),
        torch.tensor(y_eval, device=device),
    ) for vp in eval_preds], dim=0), dim=0)-torch.log(torch.tensor(num_evals))).mean()
    eval_expectation = torch.mean(torch.stack([expected_value(
        torch.tensor(vp, device=device)
    ) for vp in eval_preds], dim=0), dim=0).cpu().numpy()
    eval_rmse = ((eval_expectation-y_eval)**2).mean()**(1/2)
    
    return eval_rmse.item(), eval_nll.item()

def bnn_train(
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
        kl_coef=1,
    ):

    device = next(net.parameters()).device
    X_train_s, y_train_s, scalers = prepare_training(net, X_train, y_train)

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
            loss = -log_likelihood(preds, y_train_s[ixs], min_log_proba=-20).sum()
            loss += (1/X_train_s.shape[0]) *  bnn.get_kl_term(net) * kl_coef

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            optim.step()
            losses.append(loss.item())
        scheduler.step()
        net.eval()

        val_rmse, val_nll = bnn_eval(net, scalers, X_validation, y_validation, 20)

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


