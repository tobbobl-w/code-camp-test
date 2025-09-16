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


import numpy as np
import pandas as pd

def simulate_panel_ar1(n_individuals: int,
                       n_periods: int,
                       rho: float,
                       sigma2: float,
                       var_y0: float,
                       seed: int = None) -> pd.DataFrame:
    """
    Simulate a balanced panel AR(1) process:
        y_it = rho * y_{i,t-1} + eps_it,
    where eps_it ~ N(0, sigma2).

    Parameters
    ----------
    n_individuals : int
        Number of individuals in the panel.
    n_periods : int
        Number of time periods per individual.
    rho : float
        AR(1) persistence parameter.
    sigma2 : float
        Innovation variance of eps_it.
    var_y0 : float
        Variance of the initial observation y_{i0}.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns: ['person', 'year', 'y'].
    """
    rng = np.random.default_rng(seed)

    # Draw initial values y_{i0}
    y0 = rng.normal(loc=0.0, scale=np.sqrt(var_y0), size=n_individuals)

    # Preallocate array
    y = np.zeros((n_individuals, n_periods))
    y[:, 0] = y0

    # Iterate AR(1)
    eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=(n_individuals, n_periods - 1))
    for t in range(1, n_periods):
        y[:, t] = rho * y[:, t-1] + eps[:, t-1]

    # Return as long DataFrame
    df = pd.DataFrame({
        "person": np.repeat(np.arange(n_individuals), n_periods),
        "year": np.tile(np.arange(n_periods), n_individuals),
        "y": y.flatten()
    })
    return df


import pandas as pd
import numpy as np
import statsmodels.api as sm

def estimate_panel_ar1_statsmodels(df: pd.DataFrame,
                                   id_col: str = "person",
                                   time_col: str = "year",
                                   y_col: str = "y",
                                   add_intercept: bool = False):
    """
    Estimate AR(1): y_it = rho * y_{i,t-1} + eps_it
    using statsmodels OLS.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel data.
    id_col : str
        Column with individual IDs.
    time_col : str
        Column with time indices (must be sortable).
    y_col : str
        Column with outcome variable y.
    add_intercept : bool, default False
        If True, include a constant term in regression.

    Returns
    -------
    results : statsmodels.regression.linear_model.RegressionResultsWrapper
        Full statsmodels results object (you can call .summary()).
    """

    df = df.sort_values([id_col, time_col]).copy()
    df["y_lag"] = df.groupby(id_col)[y_col].shift(1)
    df = df.dropna(subset=["y_lag"])

    y = df[y_col]
    X = df[["y_lag"]]
    if add_intercept:
        X = sm.add_constant(X)  # adds intercept

    model = sm.OLS(y, X)
    results = model.fit()

    return results

import pandas as pd
import numpy as np
import statsmodels.api as sm

def estimate_panel_ar1_statsmodels(df: pd.DataFrame,
                                   id_col: str = "person",
                                   time_col: str = "year",
                                   y_col: str = "y",
                                   add_intercept: bool = False):
    """
    Estimate AR(1): y_it = rho * y_{i,t-1} + eps_it
    using statsmodels OLS.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel data.
    id_col : str
        Column with individual IDs.
    time_col : str
        Column with time indices (must be sortable).
    y_col : str
        Column with outcome variable y.
    add_intercept : bool, default False
        If True, include a constant term in regression.

    Returns
    -------
    results : statsmodels.regression.linear_model.RegressionResultsWrapper
        Full statsmodels results object (you can call .summary()).
    """

    df = df.sort_values([id_col, time_col]).copy()
    df["y_lag"] = df.groupby(id_col)[y_col].shift(1)
    df = df.dropna(subset=["y_lag"])

    y = df[y_col]
    X = df[["y_lag"]]
    if add_intercept:
        X = sm.add_constant(X)  # adds intercept

    model = sm.OLS(y, X)
    results = model.fit()

    return results


