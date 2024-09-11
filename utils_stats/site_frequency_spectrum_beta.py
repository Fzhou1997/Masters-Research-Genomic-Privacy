import numpy as np
import numpy.typing as npt
from scipy import stats

from .site_frequency_spectrum_normalized import site_frequency_spectrum_normalized


def site_frequency_spectrum_beta(
        alternate_allele_counts: npt.NDArray[np.int64]) -> tuple[float, float]:
    """
    Calculate the alpha and beta parameters of a Beta distribution fitted to the
    site frequency spectrum of alternate allele counts.

    This function normalizes the site frequency spectrum and fits a Beta distribution
    to the allele frequencies using the scipy.stats.beta.fit method.

    Parameters:
    alternate_allele_counts (npt.NDArray[np.int64]): An array of integers representing
                                                     the counts of alternate alleles.

    Returns:
    tuple[float, float]: A tuple containing the alpha and beta parameters of the
                         fitted Beta distribution.

    Raises:
    ValueError: If the input array is empty.
    """
    if len(alternate_allele_counts) == 0:
        raise ValueError("Alternate allele counts array must not be empty.")
    sfs = site_frequency_spectrum_normalized(alternate_allele_counts)
    allele_frequencies = np.arange(len(sfs)) / len(sfs)
    non_zero_indices = sfs > 0
    allele_frequencies = allele_frequencies[non_zero_indices]
    sfs = sfs[non_zero_indices]
    alpha, beta, loc, scale = stats.beta.fit(allele_frequencies, floc=0, fscale=1)
    return alpha, beta
