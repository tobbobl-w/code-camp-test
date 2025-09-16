"""
Tests for the camp.analysis module
"""
import pytest
import pandas as pd
import numpy as np
from camp.utils import compute_statistics
from camp.analysis import compute_moments

def test_compute_moments_basic():
    """Test basic functionality of compute_moments"""

    # Arrange: Set up test data
    # Create a simple dataset where we know the expected results

    # Test with a normal distribution
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 10000)
    result_normal = compute_moments(pd.Series(normal_data), "normal_variable")
    assert abs(result_normal['skewness']) < 0.05  # Skewness should be close to 0
    assert abs(result_normal['kurtosis'] - 3) < 0.05  # Kurtosis should be close to 3
    assert abs(result_normal['variance'] - 1) < 0.05  # Variance should be close to 1