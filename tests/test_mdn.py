import torch
from torchnaut.mdn import MDN, MDNMV


def test_mdn_shape():
    mdn = MDN(n_components=3)
    batch_size, n_components = 10, 3
    p = torch.randn(batch_size, n_components, 3)

    _ = mdn.get_dist(p)
    samples = mdn.sample(p, n=5)
    expect = mdn.expected_value(p)

    assert samples.shape == (batch_size, 5)
    assert expect.shape == (batch_size,)
    assert len(mdn.log_likelihood(p, torch.randn(batch_size))) == batch_size


def test_mdnmv_shape():
    n_dims = 2
    mdnmv = MDNMV(n_components=3, target_dim=n_dims)
    batch_size, n_components = 10, 3
    n_params = 1 + n_dims + (n_dims * (n_dims + 1) // 2)  # weights + means + tril
    p = torch.randn(batch_size, n_components * n_params)

    _ = mdnmv.get_dist(p)
    samples = mdnmv.sample(p, n=5)
    expect = mdnmv.expected_value(p)

    assert samples.shape == (batch_size, 5, n_dims)
    assert expect.shape == (batch_size, n_dims)
    assert len(mdnmv.log_likelihood(p, torch.randn(batch_size, n_dims))) == batch_size
