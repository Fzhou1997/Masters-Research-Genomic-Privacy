from typing import Any

import numpy as np
from numpy import ndarray, dtype, bool_


def likelihood_ratio_test_threshold(
        likelihood_ratio_statistics: np.ndarray[np.float64],
        labels: np.ndarray[np.bool_],
        false_positive_rate: float = 0.05) -> float:
    """
    Calculate the threshold for the likelihood ratio idash based on the given false positive rate.

    Parameters:
    likelihood_ratio_statistics (np.ndarray[np.float64]): One-dimensional array of likelihood ratio statistics.
    labels (np.ndarray[np.bool_]): One-dimensional array of boolean labels indicating positive cases.
    false_positive_rate (float): Desired false positive rate, must be between 0 and 1. Default is 0.05.

    Returns:
    float: The calculated threshold for the likelihood ratio idash.
    """
    assert likelihood_ratio_statistics.ndim == 1, "The likelihood ratio statistics must be one-dimensional."
    assert labels.ndim == 1, "The labels must be one-dimensional."
    assert likelihood_ratio_statistics.shape == labels.shape, \
        "The number of likelihood ratio statistics and labels must be the same."
    assert 0 < false_positive_rate < 1, "The false positive rate must be between 0 and 1."
    likelihood_ratio_statistics_positive = likelihood_ratio_statistics[labels]
    likelihood_ratio_statistics_negative = likelihood_ratio_statistics[~labels]
    positive_mean = np.mean(likelihood_ratio_statistics_positive)
    negative_mean = np.mean(likelihood_ratio_statistics_negative)
    if positive_mean < negative_mean:
        percentile = false_positive_rate * 100
    else:
        percentile = (1 - false_positive_rate) * 100
    threshold = np.percentile(likelihood_ratio_statistics_negative, percentile)
    return threshold


def likelihood_ratio_test(
        likelihood_ratio_statistics: np.ndarray[np.float64],
        threshold: float,
        inverted: bool = False) -> ndarray[Any, dtype[bool_]] | bool:
    """
    Perform the likelihood ratio idash using the given threshold.

    Parameters:
    likelihood_ratio_statistics (np.ndarray[np.float64]): Array of likelihood ratio statistics.
    threshold (float): Threshold value for the idash.
    inverted (bool): If True, the idash is inverted (i.e., checks for values <= threshold). Default is False.

    Returns:
    np.ndarray[np.bool_]: Boolean array indicating the idash results.
    """
    return likelihood_ratio_statistics <= threshold if inverted else likelihood_ratio_statistics >= threshold
