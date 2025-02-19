import torch
from torchnaut.utils import get_batch_ixs


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
