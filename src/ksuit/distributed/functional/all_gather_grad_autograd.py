import torch
import torch.distributed as dist


# noinspection PyAbstractClass
class AllGatherGradAutograd(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        # without the tuple call here, the gradient is not propagated for some reason
        # (therefore the backward is then not called)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients, op=dist.ReduceOp.SUM)
        grad_out = all_gradients[dist.get_rank()]
        return grad_out
