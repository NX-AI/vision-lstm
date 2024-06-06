import torch
import torch.distributed as dist


# https://discuss.pytorch.org/t/dist-all-gather-and-gradient-preservation-in-multi-gpu-training/120696/2
class AllGatherGradOverwrite:
    @staticmethod
    def apply(x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        output[dist.get_rank()] = x
        return output
