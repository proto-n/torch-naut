import torch
from torchnaut.kde import nll_gpu, nll_gpu_weighted


def test_kde_shape():
    batch_size, n_samples = 10, 100
    samples = torch.randn(batch_size, n_samples)
    y = torch.randn(batch_size, 1)

    nll = nll_gpu(samples, y)
    assert nll.shape == (batch_size,)

    w = torch.ones(batch_size, n_samples)
    w_nll = nll_gpu_weighted(samples, w, y)
    assert w_nll.shape == (batch_size,)


def test_kde_pilot_samples():
    batch_size, n_samples = 10, 100
    samples = torch.randn(batch_size, n_samples)
    w = torch.ones(batch_size, n_samples)
    y = torch.randn(batch_size, 1)

    nll1 = nll_gpu_weighted(samples, w, y, max_pilot_samples=50)
    nll2 = nll_gpu_weighted(samples, w, y, max_pilot_samples=20)

    assert nll1.shape == (batch_size,)
    assert nll2.shape == (batch_size,)
