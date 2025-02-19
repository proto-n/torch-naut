import torch
from torchnaut.crps import (
    crps_loss,
    crps_loss_weighted,
    crps_loss_mv,
    crps_loss_mv_weighted,
)
import torch.testing


def test_crps_shape():
    batch_size, n_samples = 10, 100
    yps = torch.randn(batch_size, n_samples)
    y = torch.randn(batch_size, 1)

    loss = crps_loss(yps, y)
    assert loss.shape == (batch_size,)

    w = torch.ones(batch_size, n_samples)
    w_loss = crps_loss_weighted(yps, w, y)
    torch.testing.assert_close(w_loss, loss)
    assert w_loss.shape == (batch_size,)


def test_crps_mv_shape():
    batch_size, n_samples, n_dims = 10, 100, 2
    yps = torch.randn(batch_size, n_samples, n_dims)
    y = torch.randn(batch_size, n_dims)

    loss = crps_loss_mv(yps, y)
    assert loss.shape == (batch_size,)

    w = torch.ones(batch_size, n_samples)
    w_loss = crps_loss_mv_weighted(yps, w, y)
    torch.testing.assert_close(w_loss, loss)
    assert w_loss.shape == (batch_size,)
