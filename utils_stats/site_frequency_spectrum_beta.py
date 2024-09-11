import numpy as np
import numpy.typing as npt
from scipy import stats


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
