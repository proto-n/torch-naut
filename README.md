# ![TorchNaut](https://github.com/proto-n/torch-naut/raw/main/static/naut-text.png)

## Nonparametric Aleatoric Uncertainty Modeling Toolkit for PyTorch

[![Read the Docs](https://img.shields.io/readthedocs/torch-naut?style=for-the-badge&logo=readthedocs)](https://torch-naut.readthedocs.io/en/latest/)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/proto-n/torch-naut/python-package.yml?style=for-the-badge&logo=github)](https://github.com/proto-n/torch-naut/actions/workflows/python-package.yml)

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

Official package coming soon! Right now, you can install the package as follows:
```bash
git clone https://github.com/proto-n/torch-naut
cd torch-naut
pip install .
```

## Usage

Check out the following introduction notebooks:

[1. Introduction to CRPS-based models](https://github.com/proto-n/torch-naut/blob/main/examples/1_intro_crps.ipynb)  
A full training and evaluation example of a model optimized for the CRPS loss


[2. Introduction to Mixture Density Networks](https://github.com/proto-n/torch-naut/blob/main/examples/2_intro_mdn.ipynb)  
Training and evaluating an MDN model

[3. Accounting for Epistemic Uncertainty](https://github.com/proto-n/torch-naut/blob/main/examples/3_compare_epistemic.ipynb)  
Using Deep Ensembles with CRPS-based models and MDN as the output of a Bayesian Neural Network

[4. Other architectures](https://github.com/proto-n/torch-naut/blob/main/examples/4_weighted_crps.ipynb)  
Using weighted, multihead, multilayer (in the loss sense) networks

More examples coming soon!

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
