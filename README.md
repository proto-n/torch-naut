# crps_iclr

Code for paper "Distribution-free Data Uncertainty for Neural Network Regression"

The repo contains the necessary code to reproduce the experiments.

## Setup

Reference Python (Conda) environment is given in `env.yml`.

For UCI datasets/splits and score summary, you need to clone together with `DropoutUncertaintyExps` submodule:  
```git clone --recurse-submodules --remote-submodules <repo-URL>```

Or if you cloned this one already, you can use  
```git submodule update --init --recursive```

In case you are using this without git (for example in its anonymized form during review), you can simply clone the `DropoutUncertaintyExps` repo:
```
  git clone https://github.com/yaringal/DropoutUncertaintyExps.git uci/DropoutUncertaintyExps.git
```
Some of the code uses automatic GPU selection instead of command-line arguments for device, this assumes the presence of the `nvidia-smi` command.

## Experiments

Please refer to README.md in the folder of the specific experiment.

## Structure:

- `lib`: Implementation code & utility functions
- `uci`: UCI experiments
- `multivariate`: Multivariate experiments (paraboloid & MNIST)
- `mona`: Synthetic univariate experiments (i.e., Mona Lisa)

## License

During review, we set no explicit license. After acceptance, we plan to release code under the Apache License.
