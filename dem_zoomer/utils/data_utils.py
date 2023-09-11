import pickle
from typing import Any

import numpy as np


def load_pickle(filepath: str) -> Any:
    """Load a pickle file

    Args:
        filepath (str): path to pickle file

    """
    with open(filepath, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def min_max_normalize(a: np.ndarray, dim=None) -> np.ndarray:
    """Normalize an array to [0, 1]

        If dim is not None, normalize along that dimension.
        Else scalar normalize everything

    Args:
        a (np.ndarray): array to normalize
        dim (int, optional): dimension to normalize. Defaults to None.

    Returns:
        np.ndarray: normalized array
    """
    if dim is not None:
        return (a - a.min(dim=dim, keepdims=True)) / (
            a.max(dim=dim, keepdims=True) - a.min(dim=dim, keepdims=True)
        )
    else:
        return (a - a.min()) / (a.max() - a.min())
