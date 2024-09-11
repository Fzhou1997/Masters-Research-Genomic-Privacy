import numpy as np
import numpy.typing as npt


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
