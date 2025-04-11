import torch
from sklearn.decomposition import PCA
import numpy as np


def get_batch_ixs(ref_tensor, batch_size=16, permute=False):
    """Generate batch indices for mini-batch processing.

    Args:
        ref_tensor: Reference tensor to determine total size
        batch_size: Size of each batch
        permute: Whether to randomly permute indices

    Returns:
        List of index tensors for each batch
    """
    if ref_tensor.shape[0] <= batch_size:
        return torch.arange(ref_tensor.shape[0]).unsqueeze(0)
    if permute:
        ixs = torch.randperm(ref_tensor.shape[0])
    else:
        ixs = torch.arange(ref_tensor.shape[0])
    return torch.tensor_split(ixs, ixs.shape[0] // batch_size)


class LabelScaler:
    """A scaler for preprocessing label data with optional PCA transformation.

    This class provides functionality to scale and transform label data by:
    1. Centering the data by subtracting the mean
    2. Optionally applying PCA dimensionality reduction
    3. Scaling the data to unit variance

    The transformation can be reversed using the inverse_transform method.
    """

    def __init__(self):
        """Initialize the LabelScaler."""
        pass

    def fit_transform(self, arr, pca_dims=-1, pca_whiten=True):
        """Fit the scaler to the data and transform it.

        Args:
            arr (numpy.ndarray): Input array of shape [dataset_size, *feature_dims]
            pca_dims (int): Number of PCA components. If -1, no PCA is applied
            pca_whiten (bool): Whether to apply whitening in PCA transformation

        Returns:
            numpy.ndarray: Transformed array of shape:
                - [dataset_size, prod(feature_dims)] if pca_dims=-1
                - [dataset_size, pca_dims] if PCA is applied
        """
        self.label_dims = arr.shape[1:]
        self.pca_whiten = pca_whiten
        self.pca_dims = pca_dims

        self.mean_ = np.mean(arr, axis=0, keepdims=True)
        arr = arr - self.mean_
        arr = arr.reshape(arr.shape[0], -1)
        if self.pca_dims != -1:
            self.pca = PCA(n_components=pca_dims, whiten=pca_whiten)
            arr = self.pca.fit_transform(arr)
        else:
            self.std_ = arr.std()
            return arr / arr.std()
        if self.pca_dims != -1 and not self.pca_whiten:
            self.pca_mean = arr.mean(axis=0, keepdims=True)
            self.pca_std = (
                arr.std(axis=0).max()
            )  # using max so that relative scaling is preserved, unlike with whitening
            arr = (arr - self.pca_mean) / self.pca_std
        return arr

    def transform(self, arr):
        """Transform new data using the fitted scaler.

        Args:
            arr (numpy.ndarray): Input array of shape [dataset_size, *feature_dims]
                Must match the dimensions of the data used in fit_transform

        Returns:
            numpy.ndarray: Transformed array of shape:
                - [dataset_size, prod(feature_dims)] if pca_dims=-1
                - [dataset_size, pca_dims] if PCA is applied
        """
        arr = arr - self.mean_
        if self.pca_dims != -1:
            arr = self.pca.transform(arr.reshape(arr.shape[0], -1))
        else:
            arr = arr.reshape(arr.shape[0], -1) / self.std_
        if self.pca_dims != -1 and not self.pca_whiten:
            arr = (arr - self.pca_mean) / self.pca_std
        return arr

    def inverse_transform(self, arr_scaled):
        """Inverse transform scaled data back to the original space.

        Args:
            arr_scaled (numpy.ndarray or torch.Tensor): Scaled input array of shape
                [*batch_dims, n_features] where n_features matches the output
                dimension of transform()

        Returns:
            numpy.ndarray or torch.Tensor: Array in original space with shape
                [*batch_dims, *feature_dims] where feature_dims matches the
                original input dimensions

        Example:
            If original data was shape [1000, 32, 32]:
            - Can handle inputs of shape [64, 1024] -> [64, 32, 32]
            - Can handle inputs of shape [64, 20, 1024] -> [64, 20, 32, 32]
        """
        is_tensor = isinstance(arr_scaled, torch.Tensor)
        device = arr_scaled.device if is_tensor else None
        if is_tensor:
            arr_scaled = arr_scaled.cpu().numpy()

        if self.pca_dims != -1 and not self.pca_whiten:
            arr_scaled = arr_scaled * self.pca_std + self.pca_mean

        if self.pca_dims != -1:
            arr_scaled = self.pca.inverse_transform(arr_scaled)
        else:
            arr_scaled = arr_scaled * self.std_
        arr_scaled = arr_scaled.reshape(*arr_scaled.shape[:-1], *self.label_dims)
        arr_scaled += self.mean_

        if is_tensor:
            arr_scaled = torch.from_numpy(arr_scaled).to(device)

        return arr_scaled


def calculate_pit_cdf(preds, y, weights=None):
    """Calculate the Probability Integral Transform (PIT) and its CDF.

    Args:
        preds (numpy.ndarray): Model predictions of shape [num_predictions, num_samples]
        weights (numpy.ndarray): Weights for each prediction of shape [num_predictions, num_samples]
        y (numpy.ndarray): Ground truth values of shape [num_predictions]

    Returns:
        tuple: Contains:
            - numpy.ndarray: Reference percentiles (linspace from 0 to 1)
            - numpy.ndarray: Empirical CDF of the PIT values
    """
    if weights is None:
        weights = np.ones_like(preds)

    # Sort samples for each prediction
    preds_order = np.argsort(preds, axis=1)
    all_outputs_sorted = np.take_along_axis(preds, preds_order, axis=1)
    weights_sorted = np.take_along_axis(weights, preds_order, axis=1)

    # Calculate percentiles for each prediction
    weights_to_sum = np.where(
        all_outputs_sorted < y.reshape(-1, 1),
        weights_sorted,
        np.zeros_like(weights_sorted),
    )
    percentiles = weights_to_sum.sum(axis=1) / weights.sum(axis=1)

    # Calculate the empirical cumulative distribution function (CDF) of the probability integral transformed (PIT) values
    ref_percentiles = np.linspace(0, 1, 101)
    cumulative_percentiles = np.searchsorted(
        np.sort(percentiles), ref_percentiles, side="left"
    ) / len(percentiles)

    return ref_percentiles, cumulative_percentiles
