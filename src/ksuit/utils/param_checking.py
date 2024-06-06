import collections.abc
from itertools import repeat
from pathlib import Path


# adapted from timm (timm/models/layers/helpers.py)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(repeat(x, n))

    return parse


def _is_ntuple(n):
    def check(x):
        return isinstance(x, tuple) and len(param) == n

    return check


def to_ntuple(x, n):
    return _ntuple(n=n)(x)


def is_ntuple(x, n):
    return _is_ntuple(n=n)(x)


to_2tuple = _ntuple(2)
is_2tuple = _is_ntuple(2)
to_3tuple = _ntuple(3)
is_3tuple = _is_ntuple(3)
to_4tuple = _ntuple(4)
is_4tuple = _is_ntuple(4)
to_5tuple = _ntuple(5)
is_5tuple = _is_ntuple(5)
to_6tuple = _ntuple(6)
is_6tuple = _is_ntuple(6)
to_7tuple = _ntuple(7)
is_7tuple = _is_ntuple(7)
to_8tuple = _ntuple(8)
is_8tuple = _is_ntuple(8)
to_9tuple = _ntuple(9)
is_9tuple = _is_ntuple(9)


def float_to_integer_exact(f):
    assert f.is_integer()
    return int(f)


def check_exclusive(*args):
    return sum(arg is not None for arg in args) == 1


def check_inclusive(*args):
    return sum(arg is not None for arg in args) in [0, len(args)]


def check_at_least_one(*args):
    return sum(arg is not None for arg in args) > 0


def check_at_most_one(*args):
    return sum(arg is not None for arg in args) <= 1


def check_all_none(*args):
    return sum(arg is not None for arg in args) == 0


def to_path(path, mkdir=False, should_exist=True, check_exists=True, suffix=None):
    if path is not None and not isinstance(path, Path):
        path = Path(path).expanduser()
        if mkdir:
            if not path.exists():
                path.mkdir(parents=True)
        if check_exists:
            if should_exist:
                assert path.exists(), f"'{path.as_posix()}' does not exist"
            else:
                assert not path.exists(), f"'{path.as_posix()}' already exists"
    if suffix is not None:
        assert path.as_posix().endswith(suffix), f"'{path.as_posix()}' doesnt end with '{suffix}'"
    return path


def to_list_of_values(list_or_item, default_value=None):
    if list_or_item is None:
        if default_value is None:
            return []
        else:
            return [default_value]
    if not isinstance(list_or_item, (tuple, list)):
        return [list_or_item]
    return list_or_item
