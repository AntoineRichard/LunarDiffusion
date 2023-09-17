import torch.nn.functional as F


def exists(val):
    return val is not None


def identity(t):
    return t


def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"


def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d


def cast_tuple(t, l=1):
    return ((t,) * l) if not isinstance(t, tuple) else t


def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))


def l2norm(t):
    return F.normalize(t, dim=-1)


def divisible_by(number, denom):
    return (number % denom) == 0
