import logging

import torch
from torch.cuda.amp import GradScaler

FLOAT32_ALIASES = ["float32", 32]
FLOAT16_ALIASES = ["float16", 16]
BFLOAT16_ALIASES = ["bfloat16", "bf16"]
VALID_PRECISIONS = FLOAT32_ALIASES + FLOAT16_ALIASES + BFLOAT16_ALIASES


def get_supported_precision(desired_precision, device, backup_precision=None):
    assert desired_precision in VALID_PRECISIONS
    if backup_precision is not None:
        assert backup_precision in VALID_PRECISIONS
    if desired_precision in FLOAT32_ALIASES:
        return torch.float32
    if desired_precision in FLOAT16_ALIASES:
        desired_precision = "float16"
    if desired_precision in BFLOAT16_ALIASES:
        desired_precision = "bfloat16"

    if desired_precision == "bfloat16":
        if is_bfloat16_compatible(device):
            return torch.bfloat16
        else:
            # use float16 if it is defined via backup_precision
            if backup_precision is not None and backup_precision in FLOAT16_ALIASES:
                if is_float16_compatible(device):
                    logging.info("bfloat16 not supported -> using float16")
                    return torch.float16
                else:
                    logging.info("bfloat16/float16 not supported -> using float32")
                    return torch.float32
            # use float32 as default (float16 can lead to under-/overflows)
            logging.info("bfloat16 not supported -> using float32")
            return torch.float32

    if desired_precision == "float16":
        if is_float16_compatible(device):
            return torch.float16
        else:
            # currently cpu only supports bfloat16
            if is_bfloat16_compatible(device):
                logging.info(f"float16 not supported -> using bfloat16")
                return torch.bfloat16

    logging.info(f"float16/bfloat16 not supported -> using float32")
    return torch.float32


def _is_compatible(device, dtype):
    try:
        with torch.autocast(device_type=str(device), dtype=dtype):
            pass
    except RuntimeError:
        return False
    return True


def is_bfloat16_compatible(device):
    return _is_compatible(device, torch.bfloat16)


def is_float16_compatible(device):
    return _is_compatible(device, torch.float16)


class NoopContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class NoopGradScaler:
    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def unscale_(optimizer):
        pass

    @staticmethod
    def step(optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    @staticmethod
    def update():
        pass


def get_grad_scaler_and_autocast_context(precision, device):
    if precision == torch.float32:
        return NoopGradScaler(), NoopContext()
    if precision == torch.bfloat16:
        # GradScaler shouldn't be necessary (https://github.com/pytorch/pytorch/issues/36169)
        return NoopGradScaler(), torch.autocast(str(device), dtype=precision)
    elif precision == torch.float16:
        if str(device) == "cpu":
            # GradScaler only supported for cuda/xla
            return NoopGradScaler(), torch.autocast(str(device), dtype=precision)
        else:
            return GradScaler(), torch.autocast(str(device), dtype=precision)
    raise NotImplementedError
