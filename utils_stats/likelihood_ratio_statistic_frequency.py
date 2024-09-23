import numpy as np
import numpy.typing as npt


def likelihood_ratio_statistic_frequency_diploid(
        pool_frequencies: npt.NDArray[np.float64],
        population_frequencies: npt.NDArray[np.float64],
        target_genome: npt.NDArray[np.int64],
        epsilon: float = 1e-8) -> float:
    """
    Calculate the likelihood ratio statistic for diploid organisms based on alternate allele frequencies.

    Parameters:
        pool_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the pool.
        population_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the population.
        target_genome (npt.NDArray[np.int64]): Array representing the target genome with values 0, 1, or 2.
        epsilon (float, optional): Small value to avoid taking log of and division by zero. Default is 1e-8.

    Returns:
        float: The likelihood ratio statistic.

    Raises:
        AssertionError: If pool_frequencies, population_frequencies, and target_genome do not have the same shape.
        AssertionError: If target_genome contains values other than 0, 1, or 2.
    """
    assert pool_frequencies.shape == population_frequencies.shape == target_genome.shape, \
        'pool_frequencies, population_frequencies, and target_genome must have the same shape.'
    assert np.all(np.isin(target_genome, [0, 1, 2])), 'target_genome must contain only 0, 1, or 2.'
    pool_frequencies = np.clip(pool_frequencies, epsilon, 1 - epsilon)
    population_frequencies = np.clip(population_frequencies, epsilon, 1 - epsilon)
    homozygous_reference_ratio = (1 - pool_frequencies) / (1 - population_frequencies)
    heterozygous_ratio = np.sqrt(pool_frequencies * (1 - pool_frequencies) / (population_frequencies * (1 - population_frequencies)))
    homozygous_alternate_ratio = pool_frequencies / population_frequencies
    homozygous_reference_term = (0 == target_genome) * np.log(homozygous_reference_ratio)
    heterozygous_term = (1 == target_genome) * np.log(heterozygous_ratio)
    homozygous_alternate_term = (2 == target_genome) * np.log(homozygous_alternate_ratio)
    return np.sum(homozygous_reference_term + heterozygous_term + homozygous_alternate_term)


def likelihood_ratio_statistic_frequency_haploid(
        pool_frequencies: npt.NDArray[np.float64],
        population_frequencies: npt.NDArray[np.float64],
        target_genome: npt.NDArray[np.bool_],
        epsilon: float = 1e-8) -> float:
    """
    Calculate the likelihood ratio statistic for haploid organisms based on alternate allele frequencies.

    Parameters:
        pool_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the pool.
        population_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the population.
        target_genome (npt.NDArray[np.bool_]): Array representing the target genome with boolean values.
        epsilon (float, optional): Small value to avoid taking log of and division by zero. Default is 1e-8.

    Returns:
        float: The likelihood ratio statistic.

    Raises:
        AssertionError: If pool_frequencies, population_frequencies, and target_genome do not have the same shape.
    """
    assert pool_frequencies.shape == population_frequencies.shape == target_genome.shape, \
        'pool_frequencies, population_frequencies, and target_genome must have the same shape.'
    pool_frequencies = np.clip(pool_frequencies, epsilon, 1 - epsilon)
    population_frequencies = np.clip(population_frequencies, epsilon, 1 - epsilon)
    alternate_frequency_ratio = pool_frequencies / population_frequencies
    reference_frequency_ratio = (1 - pool_frequencies) / (1 - population_frequencies)
    alternate_term = target_genome * np.log(alternate_frequency_ratio)
    reference_term = (1 - target_genome) * np.log(reference_frequency_ratio)
    return np.sum(alternate_term + reference_term)

