# ![TorchNaut](https://github.com/proto-n/torch-naut/raw/main/static/naut-text.png)

## Nonparametric Aleatoric Uncertainty Modeling Toolkit for PyTorch

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

## Coming Soon

The library code is coming soon

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
