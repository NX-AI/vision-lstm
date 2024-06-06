import numpy as np
import torch


def get_rng_from_global():
    # on dataloader worker spawn:
    # various objects require to initialize their random number generators when workers are spawned
    # the naive solution is to initialize each rng with rng = np.random.default_rng(seed=info.seed)
    # but this raises the issue that when multiple objects require an rng initialization, all rngs
    # would be seeded with info.seed
    # solution: since the numpy global rng is seeded in the worker instantiation, sample seed for rng from global rng

    # on initialization:
    # np.random.default_rng() will not be affected by np.random.set_seed (i.e. by the global numpy random seed)
    # solution: sample a random integer from np.random.randint (which is affected by np.random.set_seed)
    return np.random.default_rng(seed=np.random.randint(np.iinfo(np.int32).max))


def np_random_as_tensor(rng, eps=1e-6):
    # torch.tensor(0.9999999) results in torch.tensor(1.) but np.random can't produce 1
    tensor = torch.tensor(rng.random())
    if tensor == 1.:
        tensor -= eps
    return tensor
