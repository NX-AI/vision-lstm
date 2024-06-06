import logging
import random
from contextlib import ContextDecorator

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"set seed to {seed}")


def unset_seed():
    import time
    # current time in milliseconds
    t = 1000 * time.time()
    seed = int(t) % (2 ** 32)
    set_seed(seed)


def with_seed(seed):
    return WithSeedDecorator(seed)


class WithSeedDecorator(ContextDecorator):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        set_seed(self.seed)

    def __exit__(self, *_, **__):
        unset_seed()
