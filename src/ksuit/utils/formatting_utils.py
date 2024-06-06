import numpy as np
import torch

_SI_PREFIXES = ["", "K", "M", "G", "T", "P", "E"]


def short_number_str(number, precision=1):
    if number == 0:
        return "{short_number:.{precision}f}".format(short_number=0., precision=precision)
    if number < 0:
        number = -number
        sign = "-"
    else:
        sign = ""

    magnitude = int(np.log10(number) / 3)
    short_number = int(number / (1000 ** magnitude / 10 ** precision)) / 10 ** precision
    return "{sign}{short_number:.{precision}f}{si_unit}".format(
        sign=sign,
        short_number=short_number,
        precision=precision,
        si_unit=_SI_PREFIXES[magnitude],
    )


def summarize_indices_list(indices):
    """ [0, 1, 2, 3, 6, 7, 8] -> ["0-3", "6-8"] """
    if indices is None:
        return ["all"]
    if len(indices) == 0:
        return []
    if len(indices) == 1:
        return [str(indices[0])]
    indices = sorted(indices)
    result = []
    start_idx = end_idx = indices[0]
    for idx in indices[1:]:
        if end_idx + 1 == idx:
            end_idx = idx
        else:
            result.append(f"{start_idx}-{end_idx}" if start_idx != end_idx else str(start_idx))
            start_idx = end_idx = idx
    result.append(f"{start_idx}-{end_idx}" if start_idx != end_idx else str(start_idx))
    return result


def list_to_string(tensor):
    if torch.is_tensor(tensor):
        tensor = tensor.numpy()
    if isinstance(tensor, list):
        tensor = np.array(tensor)
    return np.array2string(tensor, precision=2, separator=", ", floatmode="fixed")


def list_to_str_without_space_and_bracket(value):
    return ",".join(str(v) for v in value)


def dict_to_string(obj, item_seperator="-"):
    """ {epoch: 5, batch_size: 64} --> epoch=5-batchsize=64 """
    assert isinstance(obj, dict)
    return item_seperator.join(f"{k}={v}" for k, v in obj.items())


def float_to_scientific_notation(value, max_precision, remove_plus=True):
    # to default scientific notation (e.g. '3.20e-06')
    float_str = "%.*e" % (max_precision, value)
    mantissa, exponent = float_str.split('e')
    # enforce precision
    mantissa = mantissa[:len("0.") + max_precision]
    # remove trailing zeros (and '.' if no zeros remain)
    mantissa = mantissa.rstrip("0").rstrip(".")
    # remove leading zeros
    exponent = f"{exponent[0]}{exponent[1:].lstrip('0')}"
    if len(exponent) == 1:
        exponent += "0"
    if remove_plus and exponent[0] == "+":
        exponent = exponent[1:]
    return f"{mantissa}e{exponent}"


def seconds_to_duration_str(total_seconds):
    tenth_milliseconds = int((total_seconds - int(total_seconds)) * 100)
    total_seconds = int(total_seconds)
    seconds = total_seconds % 60
    minutes = total_seconds % 3600 // 60
    hours = total_seconds % 86400 // 3600
    days = total_seconds // 86400
    if days > 0:
        return f"{days}-{hours:02}:{minutes:02}:{seconds:02}.{tenth_milliseconds:02}"
    return f"{hours:02}:{minutes:02}:{seconds:02}.{tenth_milliseconds:02}"
