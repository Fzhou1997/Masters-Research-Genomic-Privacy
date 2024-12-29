import numpy as np
import numpy.typing as npt
from scipy import stats


def site_frequency_spectrum(
        alternate_allele_counts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """
    Calculate the site frequency spectrum from an array of alternate allele counts.

    The site frequency spectrum is a summary of the allele frequency distribution
    in a population.

    Parameters:
    alternate_allele_counts (npt.NDArray[np.int64]): An array of integers representing
                                                     the counts of alternate alleles.

    Returns:
    npt.NDArray[np.int64]: An array where the value at each index i is the number of
                           times the count i appears in the input array.
    """
    return np.bincount(alternate_allele_counts)


def site_frequency_spectrum_beta(
        alternate_allele_counts: npt.NDArray[np.int64],
        num_samples: int) -> tuple[float, float]:
    """
    Calculate the alpha and beta parameters of the Beta distribution
    for the site frequency spectrum of alternate allele counts.

    Parameters:
    alternate_allele_counts (npt.NDArray[np.int64]): Array of counts of alternate alleles.
    num_samples (int): Total number of samples.

    Returns:
    tuple[float, float]: The alpha and beta parameters of the fitted Beta distribution.

    Raises:
    ValueError: If the alternate allele counts array is empty.
    """
    if len(alternate_allele_counts) == 0:
        raise ValueError("Alternate allele counts array must not be empty.")
    allele_frequencies = alternate_allele_counts / num_samples
    non_zero_frequencies = allele_frequencies[allele_frequencies > 0]
    alpha, beta, loc, scale = stats.beta.fit(non_zero_frequencies, floc=0, fscale=1)
    return alpha, beta


def site_frequency_spectrum_normalized(
        alternate_allele_counts: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """
    Calculate the normalized site frequency spectrum from an array of alternate allele counts.

    The site frequency spectrum is a summary of the allele frequency distribution
    in a population. The normalized site frequency spectrum is the site frequency
    spectrum divided by the total number of sites.

    Parameters:
    alternate_allele_counts (npt.NDArray[np.int64]): An array of integers representing
                                                     the counts of alternate alleles.

    Returns:
    npt.NDArray[np.float64]: An array where the value at each index i is the proportion of
                             sites with count i in the input array.
    """
    if len(alternate_allele_counts) == 0:
        return np.array([])
    sfs = site_frequency_spectrum(alternate_allele_counts)
    return sfs / len(alternate_allele_counts)