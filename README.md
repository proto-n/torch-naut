# ![TorchNaut](https://github.com/proto-n/torch-naut/raw/main/static/naut-text.png)

## Nonparametric Aleatoric Uncertainty Modeling Toolkit for PyTorch

[![Read the Docs](https://img.shields.io/readthedocs/torch-naut?style=for-the-badge&logo=readthedocs)](https://torch-naut.readthedocs.io/en/latest/)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/proto-n/torch-naut/python-package.yml?style=for-the-badge&logo=github)](https://github.com/proto-n/torch-naut/actions/workflows/python-package.yml)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/proto-n/torch-naut/python-publish.yml?style=for-the-badge&label=package)
[![PyPI - Version](https://img.shields.io/pypi/v/torchnaut?style=for-the-badge)](https://pypi.org/project/torchnaut/)

**TorchNaut** is a Python package designed for uncertainty modeling in neural networks. It provides:

- Implementations of CPRS loss-based models and Mixture Density Networks  
- Optional support for Bayesian Neural Networks and Deep Ensembles  
- GPU-accelerated adaptive-bandwidth kernel density estimation  
- Multivariate extensions of models  

TorchNaut is built as a utility library, encouraging a *bring-your-own-model* approach. However, for convenience and rapid prototyping, we also provide pre-defined models.



---

## ICLR 2025 Experiment Code

This repository was originally developed for the paper **Distribution-free Data Uncertainty for Neural Network Regression** (ICLR 2025).  
For the original, unmodified experiment code, please refer to the [ICLR2025 branch](https://github.com/proto-n/torch-naut/tree/iclr2025).

---

## Installation

You can install from PyPI using
```bash
pip install torchnaut
```

## Usage

For a *very* short intro:

Import the relevant part of the package:
```python
from torchnaut import crps
```

Extend your model with a layer that provides nondeterminism and extends variables by the sampling dimension:
```python
self.layers = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    # EpsilonSampler transforms [batch_size, 64] to [batch_size, n_samples, 64 + 16] and fills the last 16 columns with samples from the standard normal distribution.
    crps.EpsilonSampler(16), 
    nn.Linear(64 + 16, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
```

Finally, use the CRPS-loss on the model output:
```python
outputs = model(batch_X)
# CRPS loss returns a tensor of shape [batch_size, n_samples] which needs to be reduced to a scalar.
loss = crps.crps_loss(outputs, batch_y).mean()
```

However, optimal performance requires variable standardization, early stopping, etc.

For full examples, check out the following introduction notebooks:

[1. Introduction to CRPS-based models](https://github.com/proto-n/torch-naut/blob/main/examples/1_intro_crps.ipynb)  
A full training and evaluation example of a model optimized for the CRPS loss

[2. Introduction to Mixture Density Networks](https://github.com/proto-n/torch-naut/blob/main/examples/2_intro_mdn.ipynb)  
Training and evaluating an MDN model

[3. Accounting for Epistemic Uncertainty](https://github.com/proto-n/torch-naut/blob/main/examples/3_compare_epistemic.ipynb)  
Using Deep Ensembles with CRPS-based models and MDN as the output of a Bayesian Neural Network

[4. Advanced architectures](https://github.com/proto-n/torch-naut/blob/main/examples/4_weighted_crps.ipynb)  
Using weighted, multihead, multilayer (in the loss sense) networks

More examples (e.g., multivariate models) coming soon!

Also make sure to check out the [documentation](https://torch-naut.readthedocs.io/en/latest/) for an API reference.

## Citation

If you use TorchNaut in your research, please cite our paper:  
```
@inproceedings{
kelen2025distributionfree,
title={Distribution-free Data Uncertainty for Neural Network Regression},
author={Domokos M. Kelen and {\'A}d{\'a}m Jung and P{\'e}ter Kersch and Andras A Benczur},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=pDDODPtpx9}
}
```
