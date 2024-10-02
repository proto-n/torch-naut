# UCI Regression Uncertainty Benchmark

The file `run.sh` contains reference commands to reproduce the experiments presented in the paper. Results are evaluated/plotted in the files `eval_scores.ipynb` and `eval_calibration.ipynb`.

## UCI Data

For UCI datasets/splits and score summary, you need to clone together with `DropoutUncertaintyExps` submodule:  
```git clone --recurse-submodules --remote-submodules <repo-URL>```

Or if you cloned this one already, you can use  
```git submodule update --init --recursive```

In case you are using this without git (for example in its anonymized form during review), you can simply clone the `DropoutUncertaintyExps` repo to this folder:
```
  git clone https://github.com/yaringal/DropoutUncertaintyExps.git DropoutUncertaintyExps.git
```