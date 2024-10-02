# Multivariate experments

The file `run.sh` contains reference commands to reproduce the experiments presented in the paper. Results are evaluated/plotted in the files `plot_mnist.ipynb` and `plot_paraboloid.ipynb`.

## MNIST Data

For convenience, we include the MNIST dataset. To prepare the dataset for usage by our code, please first run `unzip.sh` in the `mnist_data` folder (or simply run `unzip mnist-dataset.zip`), and then execute the notebook `mnist_data/preprocess.ipynb`.

![CRPS](plots/mnist_likelihood.png "CRPS")

## Paraboloid plots

### MDN10
![MDN10](plots/paraboloid_mdn10.png "MDN10")
### MDN100
![MDN100](plots/paraboloid_mdn100.png "MDN100")
### CRPS
![CRPS](plots/paraboloid_crps.png "CRPS")
### WCRPS
![WCRPS](plots/paraboloid_wcrps.png "WCRPS")
