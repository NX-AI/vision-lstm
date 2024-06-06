import einops
import torch
import torch.distributed as dist

from .config import is_distributed, get_world_size
from .functional.all_gather_grad_autograd import AllGatherGradAutograd
from .functional.all_gather_grad_overwrite import AllGatherGradOverwrite


def get_device_and_bfloat16supported():
    # gloo cpu -> okay
    # gloo cuda -> okay (although https://pytorch.org/docs/stable/distributed.html says it isn't supported)
    # nccl cpu -> fail (but gloo anyway recommended for cpu multiprocessing)
    # nccl cuda -> okay
    # bfloat16 cpu -> fail
    if not is_distributed():
        return torch.device("cpu"), True
    if dist.get_backend() == "nccl":
        return torch.device("cuda"), True
    if dist.get_backend() == "gloo":
        return torch.device("cpu"), False
    raise NotImplementedError


def get_bool_gather_supported():
    if not is_distributed():
        return True
    if dist.get_backend() == "nccl":
        return True
    if dist.get_backend() == "gloo":
        return False
    raise NotImplementedError


def _prepare_tensor(x):
    """
    prepare for distributed communication
    - wrap primitive types into tensors
    - push tensor onto supported device
    - convert bool to float if bool gathering is not supported
    - call .contiguous if x is not in a contiguous memory block
    """
    device, bfloat16_supported = get_device_and_bfloat16supported()
    # I think this doesn't work in some configuration not sure in which though
    # note in which configuration and convert back to bool after gather
    if isinstance(x, bool):
        # x = torch.tensor(x, dtype=torch.float32, device=device)
        # og_device = torch.device("cpu")
        raise RuntimeError
    if isinstance(x, (float, int, list, tuple)):
        x = torch.tensor(x, device=device)
        og_device = torch.device("cpu")
    else:
        og_device = x.device
    if x.dtype == torch.bfloat16 and not bfloat16_supported:
        x = x.type(torch.float32)
    # bool gather is not supported in some settings
    if x.dtype == torch.bool and not get_bool_gather_supported():
        x = x.type(torch.float32)
        to_bool = True
    else:
        to_bool = False
    if not x.is_contiguous():
        x = x.contiguous()
    return x.to(device), og_device, to_bool


def _all_gather_grad(x, all_gather_fn, batch_dim=0):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        result = all_gather_fn(x)
        if result[0].ndim == 0:
            # scalars can't be concatenated
            result = [r.unsqueeze(0) for r in result]
        result = torch.concat(result, dim=batch_dim).to(og_device)
    else:
        result = _all_gather_nondistributed(x, og_device)
    if to_bool:
        result = result.bool()
    return result


def all_gather_grad(x, batch_dim=0):
    return _all_gather_grad(x, AllGatherGradAutograd.apply, batch_dim=batch_dim)
    # return _all_gather_grad(x, AllGatherGradOverwrite.apply)


def all_gather_grad_autograd(x):
    return _all_gather_grad(x, AllGatherGradAutograd.apply)


def all_gather_grad_overwrite(x):
    return _all_gather_grad(x, AllGatherGradOverwrite.apply)


@torch.no_grad()
def all_gather_nograd(x):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        result = [torch.zeros_like(x) for _ in range(get_world_size())]
        dist.all_gather(result, x)
        if result[0].ndim == 0:
            # scalars can't be concatenated
            result = torch.tensor(result, device=og_device)
        else:
            result = torch.concat(result).to(og_device)
    else:
        result = _all_gather_nondistributed(x, og_device).detach()
    if to_bool:
        result = result.bool()
    return result


def _all_gather_nondistributed(x, og_device):
    if x.ndim == 0:
        # distributed gather adds a dimension to scalars
        x = x.unsqueeze(0)
    return x.to(og_device)


def all_gather_nograd_clipped(x, max_length):
    result = all_gather_nograd(x)
    if is_distributed():
        # gathering changes the order of the samples -> correct them
        # most of the time this is not noeeded (e.g. for metrics) as the order is not important
        # for things like predictions it does matter
        # 1 GPU: [0, 1, 2, 3, 4, 5, 6, 7]
        # 2 GPU: [0, 2, 4, 6] + [1, 3, 5, 7]
        # 4 GPU: [0, 4] + [1, 5] + [2, 6] + [3, 7]
        result = einops.rearrange(
            result,
            "(num_gpus len_per_gpu) ... -> (len_per_gpu num_gpus) ...",
            num_gpus=get_world_size(),
        )
        # DistributedSampler pads the dataset to give every GPU the same amount of samples
        return result[:max_length]
    return result


def all_reduce_sum_nograd(x):
    with torch.no_grad():
        return all_reduce_sum_grad(x)


def all_reduce_sum_grad(x):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        # all_reduce is differentiable https://github.com/pytorch/pytorch/issues/58005
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


def all_reduce_mean_grad(x):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        x = all_reduce_sum_grad(x) / get_world_size()
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


def all_reduce_mean_nograd(x):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        x = all_reduce_sum_nograd(x) / get_world_size()
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x
