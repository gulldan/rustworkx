# ruff: noqa: E501
import math
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np


def measure_memory(func: Callable[..., Any]) -> Callable[..., tuple[Any, float]]:
    """Decorator to measure peak memory usage of a function.

    Uses `tracemalloc` to track memory allocation.

    Args:
        func: The function to measure.

    Returns:
        A wrapper function that, when called, executes the original function
        and returns a tuple containing the original function's result and
        the peak memory usage in megabytes (MB).
    """

    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        tracemalloc.start()
        result: Any = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()  # current, peak
        tracemalloc.stop()
        return result, peak / (1024 * 1024)  # Convert to MB

    return wrapper


def format_memory(memory_mb: float) -> str:
    """Formats memory usage with adaptive units (KB, MB, GB).

    Displays the memory value with a maximum of 3 significant digits.

    Args:
        memory_mb: Memory usage in megabytes (MB).

    Returns:
        A string representing the formatted memory usage with an
        appropriate unit (KB, MB, or GB).
    """
    if memory_mb < 0.1:
        memory_kb: float = memory_mb * 1024
        value: float = (
            round(memory_kb, 3 - int(math.floor(math.log10(abs(memory_kb)))) - 1)
            if memory_kb >= 1 and memory_kb != 0
            else memory_kb
        )
        return f"{value:.3g} KB"
    elif memory_mb < 1000:
        value = (
            round(memory_mb, 3 - int(math.floor(math.log10(abs(memory_mb)))) - 1)
            if memory_mb >= 1 and memory_mb != 0
            else memory_mb
        )
        return f"{value:.3g} MB"
    else:
        memory_gb: float = memory_mb / 1024
        value = (
            round(memory_gb, 3 - int(math.floor(math.log10(abs(memory_gb)))) - 1)
            if memory_gb >= 1 and memory_gb != 0
            else memory_gb
        )
        return f"{value:.3g} GB"


def format_time(seconds: float) -> str:
    """Formats time duration with adaptive units (µs, ms, s).

    Displays the time value with a maximum of 3 significant digits.

    Args:
        seconds: Time duration in seconds.

    Returns:
        A string representing the formatted time duration with an
        appropriate unit (µs, ms, or s).
    """
    if seconds == 0:
        return "0 s"

    microseconds: float = seconds * 1000000
    value: float

    if microseconds < 1000:  # μs range
        value = (
            round(microseconds, 3 - int(math.floor(math.log10(abs(microseconds)))) - 1)
            if microseconds >= 1 and microseconds != 0
            else microseconds
        )
        return f"{value:.3g} μs"
    elif microseconds < 1000000:  # ms range
        milliseconds: float = microseconds / 1000
        value = (
            round(milliseconds, 3 - int(math.floor(math.log10(abs(milliseconds)))) - 1)
            if milliseconds >= 1 and milliseconds != 0
            else milliseconds
        )
        return f"{value:.3g} ms"
    else:  # s range
        value = (
            round(seconds, 3 - int(math.floor(math.log10(abs(seconds)))) - 1)
            if seconds >= 1 and seconds != 0
            else seconds
        )
        return f"{value:.3g} s"


def format_bool(value: Any) -> str:
    """Formats bool-like exact-match values for tables.

    Args:
        value: A bool/number/string/None value.

    Returns:
        "Yes", "No", or "N/A".
    """
    if value is None:
        return "N/A"
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int | float):
        return "Yes" if float(value) != 0.0 else "No"
    if isinstance(value, str):
        value_norm = value.strip().lower()
        if value_norm in {"true", "yes", "y", "1"}:
            return "Yes"
        if value_norm in {"false", "no", "n", "0"}:
            return "No"
    return "N/A"
