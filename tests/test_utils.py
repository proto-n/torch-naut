import torch
from torchnaut.utils import get_batch_ixs
import numpy as np
from torchnaut.utils import LabelScaler
import pytest


def test_batch_ixs():
    tensor = torch.randn(100, 10)

    # Test without permutation
    batches = get_batch_ixs(tensor, batch_size=16, permute=False)
    assert len(batches) == 6  # 100/16 rounded down
    assert all(isinstance(b, torch.Tensor) for b in batches)
    assert sum([len(b) for b in batches]) == 100

    # Test with permutation
    batches = get_batch_ixs(tensor, batch_size=16, permute=True)
    assert len(batches) == 6
    assert all(isinstance(b, torch.Tensor) for b in batches)
    assert sum([len(b) for b in batches]) == 100

    # Test small batch
    small_tensor = torch.randn(10, 10)
    batches = get_batch_ixs(small_tensor, batch_size=16)
    assert len(batches) == 1
    assert sum([len(b) for b in batches]) == 10


def test_label_scaler_basic():
    # Test basic scaling without PCA
    scaler = LabelScaler()
    data = np.random.randn(100, 5)

    # Test fit_transform
    transformed = scaler.fit_transform(data, pca_dims=-1)
    assert transformed.shape == (100, 5)
    assert np.abs(transformed.mean()) < 1e-6
    assert np.abs(transformed.std() - 1.0) < 1e-6

    # Test transform
    new_data = np.random.randn(20, 5)
    transformed_new = scaler.transform(new_data)
    assert transformed_new.shape == (20, 5)

    # Test inverse_transform
    reconstructed = scaler.inverse_transform(transformed)
    assert reconstructed.shape == data.shape
    np.testing.assert_array_almost_equal(reconstructed, data, decimal=6)


def test_label_scaler_pca():
    scaler = LabelScaler()
    data = np.random.randn(100, 10)

    # Test PCA with whitening
    transformed = scaler.fit_transform(data, pca_dims=8, pca_whiten=True)
    assert transformed.shape == (100, 8)
    assert np.abs(transformed.mean()) < 1e-2
    print(transformed.std())
    assert np.abs(transformed.std() - 1.0) < 1e-2

    # Test PCA without whitening
    scaler2 = LabelScaler()
    transformed2 = scaler2.fit_transform(data, pca_dims=8, pca_whiten=False)
    assert transformed2.shape == (100, 8)


def test_label_scaler_tensor_input():
    scaler = LabelScaler()
    data = np.random.randn(100, 5)
    transformed = scaler.fit_transform(data)

    # Test with torch tensor input
    tensor_data = torch.from_numpy(transformed).cuda()
    reconstructed = scaler.inverse_transform(tensor_data)

    assert isinstance(reconstructed, torch.Tensor)
    assert reconstructed.device.type == "cuda"
    assert reconstructed.shape == (100, 5)

    # Compare with numpy reconstruction
    numpy_reconstructed = scaler.inverse_transform(transformed)
    torch.testing.assert_close(reconstructed.cpu().numpy(), numpy_reconstructed)


def test_label_scaler_multidim():
    scaler = LabelScaler()
    # Test with 3D data (e.g., images)
    data = np.random.randn(100, 16, 16)

    transformed = scaler.fit_transform(data)
    assert transformed.shape[0] == 100

    reconstructed = scaler.inverse_transform(transformed)
    assert reconstructed.shape == data.shape
    np.testing.assert_array_almost_equal(reconstructed, data, decimal=6)


@pytest.mark.parametrize("shape", [(100, 5), (100, 16, 16), (100, 3, 32, 32)])
def test_label_scaler_shapes(shape):
    scaler = LabelScaler()
    data = np.random.randn(*shape)

    # Test without PCA
    transformed = scaler.fit_transform(data, pca_dims=-1)
    reconstructed = scaler.inverse_transform(transformed)
    assert reconstructed.shape == data.shape

    if len(shape) == 2:  # Only test PCA with 2D data
        # Test with PCA
        scaler_pca = LabelScaler()
        transformed_pca = scaler_pca.fit_transform(data, pca_dims=min(shape[1], 3))
        reconstructed_pca = scaler_pca.inverse_transform(transformed_pca)
        assert reconstructed_pca.shape == data.shape


def test_label_scaler_sample_dimension():
    # Test with 1D labels that get expanded in inverse transform
    scaler = LabelScaler()

    # Original data: [dataset_size, 1]
    data = np.random.randn(100, 1)
    transformed = scaler.fit_transform(data)
    assert transformed.shape == (100, 1)

    # Create batch with multiple samples: [batch_size, num_samples, 1]
    batch_size, num_samples = 10, 20
    sample_data = torch.randn(batch_size, num_samples, 1)
    reconstructed = scaler.inverse_transform(sample_data)
    assert reconstructed.shape == (batch_size, num_samples, 1)

    # Test with higher dimensional data
    data_3d = np.random.randn(100, 3)  # [dataset_size, 3]
    transformed_3d = scaler.fit_transform(data_3d)
    assert transformed_3d.shape == (100, 3)

    # Inverse transform with samples: [batch_size, num_samples, 3]
    sample_data_3d = torch.randn(batch_size, num_samples, 3)
    reconstructed_3d = scaler.inverse_transform(sample_data_3d)
    assert reconstructed_3d.shape == (batch_size, num_samples, 3)


def test_label_scaler_complex_shapes():
    scaler = LabelScaler()

    # Test with image-like labels: [dataset_size, channels, height, width]
    data = np.random.randn(1000, 3, 32, 32)
    transformed = scaler.fit_transform(data)
    # Transformed shape should be [dataset_size, flattened_features]
    assert transformed.shape == (1000, 3 * 32 * 32)

    # Test case 1: Simple batch dimension
    # Input shape: [batch_size, flattened_features]
    batch_data = torch.randn(64, 3 * 32 * 32)
    reconstructed = scaler.inverse_transform(batch_data)
    assert reconstructed.shape == (64, 3, 32, 32)

    # Test case 2: Multiple batch dimensions
    # Input shape: [batch_size, num_samples, flattened_features]
    sample_data = torch.randn(64, 20, 3 * 32 * 32)
    reconstructed = scaler.inverse_transform(sample_data)
    assert reconstructed.shape == (64, 20, 3, 32, 32)
