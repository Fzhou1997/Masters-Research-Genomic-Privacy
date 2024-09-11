import numpy as np
import numpy.typing as npt

from .site_frequency_spectrum import site_frequency_spectrum


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
