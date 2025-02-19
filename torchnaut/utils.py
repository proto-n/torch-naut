import torch


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
