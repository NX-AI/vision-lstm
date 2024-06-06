import torch
import torch.distributed as dist


def cuda_start_event():
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    return start_event


def cuda_end_event(start_event):
    if dist.is_available() and dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    torch.cuda.synchronize()
    # torch.cuda.Event.elapsed_time returns milliseconds but kappaprofiler expects seconds
    return start_event.elapsed_time(end_event) / 1000
