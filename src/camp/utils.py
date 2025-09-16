from typing import Sequence
from scipy.stats import skew, kurtosis
from typing import Dict, Union, Optional
import numpy as np

def hello(name:str) -> float:
    print(f"Hello again one more time, {name}!")
    return 0.0

def compute_mean(numbers: Sequence[float]) -> float:
    assert len(numbers) > 0, "The input sequence must not be empty."
    assert all( numbers != 0.0 for numbers in numbers), "All numbers must be non-zero."
    return sum(numbers) / len(numbers)


def compute_statistics(arr: np.ndarray) -> dict:
    """
    Compute variance, skewness, and kurtosis of a numpy array.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        dict: Dictionary with 'variance', 'skewness', and 'kurtosis'.
    """
    variance = np.var(arr, ddof=1)
    skewness = skew(arr)
    kurt = kurtosis(arr)
    return {
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurt
    }

