"""
Tests for the camp.analysis module
"""
import pytest
import pandas as pd
import numpy as np
from camp.utils import compute_statistics

def test_compute_moments_basic():
    """Test basic functionality of compute_moments"""

    # Arrange: Set up test data
    # Create a simple dataset where we know the expected results

    # Test with a normal distribution
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 10000)
    result_normal = compute_statistics(normal_data)
    assert abs(result_normal['skewness']) < 0.05  # Skewness should be close to 0
    assert abs(result_normal['kurtosis']) < 0.05  # Kurtosis should be close to 3
    assert abs(result_normal['variance'] - 1) < 0.05  # Variance should be close to 1


def test_compute_statistics_simple_array():
    """Test compute_statistics with a simple known array"""
    # Test with array [1, 2, 3, 4, 5]
    data = np.array([1, 2, 3, 4, 5])
    result = compute_statistics(data)

    # Expected variance (sample variance with ddof=1)
    expected_variance = 2.5
    assert abs(result['variance'] - expected_variance) < 1e-10

    # For symmetric data, skewness should be 0
    assert abs(result['skewness']) < 1e-10

    # Check that all expected keys are present
    assert 'variance' in result
    assert 'skewness' in result
    assert 'kurtosis' in result


def test_compute_statistics_constant_array():
    """Test compute_statistics with constant values"""
    data = np.array([5.0, 5.0, 5.0, 5.0])
    result = compute_statistics(data)

    # Variance should be 0 for constant data
    assert result['variance'] == 0.0
    # Skewness and kurtosis are undefined for constant data but scipy returns nan
    assert np.isnan(result['skewness'])
    assert np.isnan(result['kurtosis'])


def test_compute_statistics_single_value():
    """Test compute_statistics with single value array"""
    data = np.array([42.0])
    result = compute_statistics(data)

    # Variance is undefined for single value with ddof=1, should be nan
    assert np.isnan(result['variance'])
    assert np.isnan(result['skewness'])
    assert np.isnan(result['kurtosis'])


def test_compute_statistics_negative_values():
    """Test compute_statistics with negative values"""
    data = np.array([-5, -3, -1, 1, 3, 5])
    result = compute_statistics(data)

    # Should handle negative values correctly
    assert isinstance(result['variance'], float)
    assert isinstance(result['skewness'], float)
    assert isinstance(result['kurtosis'], float)

    # Symmetric data should have skewness close to 0
    assert abs(result['skewness']) < 1e-10


def test_compute_statistics_skewed_data():
    """Test compute_statistics with positively skewed data"""
    # Create positively skewed data
    data = np.array([1, 1, 1, 1, 1, 2, 2, 3, 10])
    result = compute_statistics(data)

    # Should have positive skewness
    assert result['skewness'] > 0
    assert result['variance'] > 0


def test_compute_statistics_empty_array():
    """Test compute_statistics with empty array"""
    data = np.array([])
    result = compute_statistics(data)

    # All statistics should be NaN for empty array
    assert np.isnan(result['variance'])
    assert np.isnan(result['skewness'])
    assert np.isnan(result['kurtosis'])


def test_compute_statistics_return_type():
    """Test that compute_statistics returns correct data types"""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_statistics(data)

    assert isinstance(result, dict)
    assert len(result) == 3

    for key in ['variance', 'skewness', 'kurtosis']:
        assert key in result
        assert isinstance(result[key], (float, np.floating)) or np.isnan(result[key])


def test_compute_statistics_large_values():
    """Test compute_statistics with large numerical values"""
    data = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    result = compute_statistics(data)

    # Should handle large values without overflow
    assert np.isfinite(result['variance'])
    assert np.isfinite(result['skewness'])
    assert np.isfinite(result['kurtosis'])