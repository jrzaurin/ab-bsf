"""Small shared helpers."""

import functools
import inspect
from typing import Callable, Dict, List, Union

import numpy as np

__all__ = ["alias_params", "check_list_lengths", "safe_divide"]


def alias_params(**aliases: Union[str, List[str]]) -> Callable:
    """
    Let a function accept alternative names for some of its parameters.

    ``@alias_params(p1="mean1", n1=["n_control", "initial_control_sample_size"])``
    means a caller can write ``mean1=100`` instead of ``p1=100``. Passing both the
    canonical name and an alias, or two aliases of the same parameter, raises.

    Examples
    --------
    >>> @alias_params(p2="mean2")
    ... def f(p2):
    ...     return p2
    >>> f(mean2=3)
    3
    """
    alias_to_canonical: Dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in [names] if isinstance(names, str) else names:
            alias_to_canonical[name] = canonical

    def decorator(func: Callable) -> Callable:
        params = inspect.signature(func).parameters
        for canonical in aliases:
            if canonical not in params:
                raise TypeError(
                    f"{func.__name__} has no parameter '{canonical}' to alias"
                )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for alias, canonical in alias_to_canonical.items():
                if alias in kwargs:
                    if canonical in kwargs:
                        raise TypeError(
                            f"{func.__name__}() got both '{canonical}' and its alias '{alias}'"
                        )
                    kwargs[canonical] = kwargs.pop(alias)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_list_lengths(lists: List[List]) -> None:
    """Raise if the given lists do not all have the same length."""
    it = iter(lists)
    the_len = len(next(it))
    if not all(len(i) == the_len for i in it):
        raise ValueError("Not all lists have same length!")


def safe_divide(num: float, den: float) -> float:
    """num / den, or inf/nan when den is zero."""
    if den == 0:
        return float("nan") if num == 0 else float(np.sign(num) * np.inf)
    return num / den
