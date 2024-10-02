import torch
import numpy as np
import subprocess
from copy import deepcopy

def get_batch_ixs(like_tensor, batch_size=16, permute=False):
    if(like_tensor.shape[0] <= batch_size):
        return torch.arange(like_tensor.shape[0]).unsqueeze(0)
    if permute:
        ixs = torch.randperm(like_tensor.shape[0])
    else:
        ixs = torch.arange(like_tensor.shape[0])
    return torch.tensor_split(ixs, ixs.shape[0]//batch_size)


def select_gpu_with_low_usage():
    device_ids = list(range(torch.cuda.device_count()))
    min_memory_free = 20000
    min_gpu_id = -1

    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    gpu_memory = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]

    for gpu_id in device_ids:
        torch.cuda.set_device(gpu_id)
        memory_usage = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        gpu_utilization = torch.cuda.utilization(gpu_id)
        if gpu_memory[gpu_id] > min_memory_free and gpu_utilization < 10:
            min_memory_usage = memory_usage
            min_gpu_id = gpu_id

    return min_gpu_id