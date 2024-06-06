import numpy as np


def get_powers_of_two(min_value, max_value):
    powers_of_two = []
    if max_value > 0:
        powers_of_two += [2 ** i for i in range(int(np.log2(max_value)) + 1)]
    return [p for p in powers_of_two if p >= min_value]


def is_power_of_two(value):
    return np.log2(value).is_integer()
