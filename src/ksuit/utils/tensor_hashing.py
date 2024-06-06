import numpy as np
import torch

PRIMES = [
    957599,
    115459,
    635809,
    595087,
    542771,
    198733,
    180361,
    168109,
    963667,
    623869,
    220469,
    252359,
    649799,
    502717,
    520837,
    924601,
    909697,
    341743,
    675029,
    609173,
    671357,
    269419,
    283051,
    703223,
    409987,
    132361,
    215617,
    824477,
    155153,
    599611,
    654541,
    731041,
    346651,
    840473,
    961991,
    596579,
    798487,
    216973,
    206593,
    924719,
    358223,
    472721,
    181549,
    859561,
    238237,
    174533,
    278561,
    876203,
    610031,
    716953,
]


def hash_tensor_entries(tensor, num_primes=None, shuffle_primes_seed=None):
    if num_primes is None:
        num_primes = 10
    primes = PRIMES
    if shuffle_primes_seed is not None:
        primes = np.random.default_rng(seed=shuffle_primes_seed).permutation(primes)
    primes = primes[:num_primes]
    for i in range(len(primes) - 2):
        tensor = (tensor + primes[i]) * primes[i + 1] % primes[i + 2]
    return tensor


def hash_rgb(tensor, dim=1):
    r = hash_tensor_entries(tensor, shuffle_primes_seed=0) % 255
    g = hash_tensor_entries(tensor, shuffle_primes_seed=1) % 255
    b = hash_tensor_entries(tensor, shuffle_primes_seed=2) % 255
    tensor = torch.stack([r, g, b], dim=dim) / 255
    return tensor
