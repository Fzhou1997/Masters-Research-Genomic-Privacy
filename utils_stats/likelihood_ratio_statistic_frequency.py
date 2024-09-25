import numpy as np
import numpy.typing as npt


def likelihood_ratio_statistic_frequency_diploid(
        pool_frequencies: npt.NDArray[np.float64],
        population_frequencies: npt.NDArray[np.float64],
        target_genomes: npt.NDArray[np.int64],
        epsilon: float = 1e-8) -> float | npt.NDArray[np.float64]:
    """
    Calculate the likelihood ratio statistic for diploid organisms based on alternate allele frequencies.

    Parameters:
        pool_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the pool.
        population_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the population.
        target_genomes (npt.NDArray[np.int64]): Array representing the target genomes with values 0, 1, or 2.
        epsilon (float, optional): Small value to avoid taking log of and division by zero. Default is 1e-8.

    Returns:
        float | npt.NDArray[np.float64]: The likelihood ratio statistics.

    Raises:
        AssertionError: If pool_frequencies and population_frequencies do not have the same shape.
        AssertionError: If target_genomes is not one-dimensional or two-dimensional.
        AssertionError: If pool_frequencies and target_genomes do not have the same shape.
        AssertionError: If the first dimension of pool_frequencies is not the same as the second dimension of target_genomes.
        AssertionError: If target_genomes does not contain only 0, 1, or 2.
    """
    assert pool_frequencies.ndim == 1, \
        'pool_frequencies must be one-dimensional.'
    assert population_frequencies.ndim == 1, \
        'population_frequencies must be one-dimensional.'
    assert pool_frequencies.shape == population_frequencies.shape, \
        'pool_frequencies and population_frequencies must have the same shape.'
    assert target_genomes.ndim == 1 or target_genomes.ndim == 2, \
        'target_genome must be one-dimensional or two-dimensional.'
    if target_genomes.ndim == 1:
        assert pool_frequencies.shape == target_genomes.shape, \
            'pool_frequencies and target_genome must have the same shape.'
    else:
        assert pool_frequencies.shape[0] == target_genomes.shape[1], \
            'The first dimension of pool_frequencies must be the same as the second dimension of target_genomes.'
    assert np.all(np.isin(target_genomes, [0, 1, 2])), 'target_genome must contain only 0, 1, or 2.'
    pool_frequencies = np.clip(pool_frequencies, epsilon, 1 - epsilon)
    population_frequencies = np.clip(population_frequencies, epsilon, 1 - epsilon)
    homozygous_reference_ratio = (1 - pool_frequencies) / (1 - population_frequencies)
    heterozygous_ratio = np.sqrt(pool_frequencies * (1 - pool_frequencies) / (population_frequencies * (1 - population_frequencies)))
    homozygous_alternate_ratio = pool_frequencies / population_frequencies
    homozygous_reference_term = (0 == target_genomes) * np.log(homozygous_reference_ratio)
    heterozygous_term = (1 == target_genomes) * np.log(heterozygous_ratio)
    homozygous_alternate_term = (2 == target_genomes) * np.log(homozygous_alternate_ratio)
    return np.sum(homozygous_reference_term + heterozygous_term + homozygous_alternate_term, axis=1)


def likelihood_ratio_statistic_frequency_haploid(
        pool_frequencies: npt.NDArray[np.float64],
        population_frequencies: npt.NDArray[np.float64],
        target_genomes: npt.NDArray[np.bool_],
        epsilon: float = 1e-8) -> float | npt.NDArray[np.float64]:
    """
    Calculate the likelihood ratio statistic for haploid organisms based on alternate allele frequencies.

    Parameters:
        pool_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the pool.
        population_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the population.
        target_genomes (npt.NDArray[np.bool_]): Array representing the target genome with boolean values.
        epsilon (float, optional): Small value to avoid taking log of and division by zero. Default is 1e-8.

    Returns:
        float | npt.NDArray[np.float64]: The likelihood ratio statistics.

    Raises:
        AssertionError: If pool_frequencies and population_frequencies do not have the same shape.
        AssertionError: If target_genomes is not one-dimensional or two-dimensional.
        AssertionError: If pool_frequencies and target_genomes do not have the same shape.
        AssertionError: If the first dimension of pool_frequencies is not the same as the second dimension of target_genomes.
    """
    assert pool_frequencies.ndim == 1, \
        'pool_frequencies must be one-dimensional.'
    assert population_frequencies.ndim == 1, \
        'population_frequencies must be one-dimensional.'
    assert pool_frequencies.shape == population_frequencies.shape, \
        'pool_frequencies and population_frequencies must have the same shape.'
    assert target_genomes.ndim == 1 or target_genomes.ndim == 2, \
        'target_genomes must be one-dimensional or two-dimensional.'
    if target_genomes.ndim == 1:
        assert pool_frequencies.shape == target_genomes.shape, \
            'pool_frequencies and target_genome must have the same shape.'
    else:
        assert pool_frequencies.shape[0] == target_genomes.shape[1], \
            'The first dimension of pool_frequencies must be the same as the second dimension of target_genomes.'
    pool_frequencies = np.clip(pool_frequencies, epsilon, 1 - epsilon)
    population_frequencies = np.clip(population_frequencies, epsilon, 1 - epsilon)
    alternate_frequency_ratio = pool_frequencies / population_frequencies
    reference_frequency_ratio = (1 - pool_frequencies) / (1 - population_frequencies)
    alternate_term = target_genomes * np.log(alternate_frequency_ratio)
    reference_term = (1 - target_genomes) * np.log(reference_frequency_ratio)
    return np.sum(alternate_term + reference_term, axis=1)


def likelihood_ratio_statistic_frequency_diploid_approximate(
        pool_frequencies: npt.NDArray[np.float64],
        reference_frequencies: npt.NDArray[np.float64],
        target_genomes: npt.NDArray[np.int64],
        epsilon: float = 1e-8) -> float | npt.NDArray[np.float64]:
    """
    Calculate the likelihood ratio statistic for diploid organisms based on alternate allele frequencies using an approximate method.

    Parameters:
        pool_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the pool.
        reference_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the reference population.
        target_genomes (npt.NDArray[np.int64]): Array representing the target genomes with values 0, 1, or 2.
        epsilon (float, optional): Small value to avoid taking log of and division by zero. Default is 1e-8.

    Returns:
        float | npt.NDArray[np.float64]: The likelihood ratio statistics.

    Raises:
        AssertionError: If pool_frequencies and reference_frequencies do not have the same shape.
        AssertionError: If target_genomes is not one-dimensional or two-dimensional.
        AssertionError: If pool_frequencies and target_genomes do not have the same shape.
        AssertionError: If the first dimension of pool_frequencies is not the same as the second dimension of target_genomes.
        AssertionError: If target_genomes does not contain only 0, 1, or 2.
    """
    assert pool_frequencies.ndim == 1, \
        'pool_frequencies must be one-dimensional.'
    assert reference_frequencies.ndim == 1, \
        'reference_frequencies must be one-dimensional.'
    assert pool_frequencies.shape == reference_frequencies.shape, \
        'pool_frequencies and reference_frequencies must have the same shape.'
    assert target_genomes.ndim == 1 or target_genomes.ndim == 2, \
        'target_genome must be one-dimensional or two-dimensional.'
    if target_genomes.ndim == 1:
        assert pool_frequencies.shape == target_genomes.shape, \
            'pool_frequencies and target_genome must have the same shape.'
    else:
        assert pool_frequencies.shape[0] == target_genomes.shape[1], \
            'The first dimension of pool_frequencies must be the same as the second dimension of target_genomes.'
    assert np.all(np.isin(target_genomes, [0, 1, 2])), 'target_genome must contain only 0, 1, or 2.'
    pool_frequencies = np.clip(pool_frequencies, epsilon, 1 - epsilon)
    reference_frequencies = np.clip(reference_frequencies, epsilon, 1 - epsilon)
    homozygous_reference_ratio = (1 - pool_frequencies) / (1 - reference_frequencies)
    heterozygous_ratio = np.sqrt(pool_frequencies * (1 - pool_frequencies) / (reference_frequencies * (1 - reference_frequencies)))
    homozygous_alternate_ratio = pool_frequencies / reference_frequencies
    homozygous_reference_term = (0 == target_genomes) * np.log(homozygous_reference_ratio)
    heterozygous_term = (1 == target_genomes) * np.log(heterozygous_ratio)
    homozygous_alternate_term = (2 == target_genomes) * np.log(homozygous_alternate_ratio)
    return np.sum(homozygous_reference_term + heterozygous_term + homozygous_alternate_term, axis=1)


def likelihood_ratio_statistic_frequency_haploid_approximate(
        pool_frequencies: npt.NDArray[np.float64],
        reference_frequencies: npt.NDArray[np.float64],
        target_genomes: npt.NDArray[np.bool_],
        epsilon: float = 1e-8) -> float | npt.NDArray[np.float64]:
    """
    Calculate the likelihood ratio statistic for haploid organisms based on alternate allele frequencies using an approximate method.

    Parameters:
        pool_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the pool.
        reference_frequencies (npt.NDArray[np.float64]): Array of alternate allele frequencies in the reference population.
        target_genomes (npt.NDArray[np.bool_]): Array representing the target genomes with boolean values.
        epsilon (float, optional): Small value to avoid taking log of and division by zero. Default is 1e-8.

    Returns:
        float | npt.NDArray[np.float64]: The likelihood ratio statistics.

    Raises:
        AssertionError: If pool_frequencies and reference_frequencies do not have the same shape.
        AssertionError: If target_genomes is not one-dimensional or two-dimensional.
        AssertionError: If pool_frequencies and target_genomes do not have the same shape.
        AssertionError: If the first dimension of pool_frequencies is not the same as the second dimension of target_genomes.
    """
    assert pool_frequencies.ndim == 1, \
        'pool_frequencies must be one-dimensional.'
    assert reference_frequencies.ndim == 1, \
        'reference_frequencies must be one-dimensional.'
    assert pool_frequencies.shape == reference_frequencies.shape, \
        'pool_frequencies and reference_frequencies must have the same shape.'
    assert target_genomes.ndim == 1 or target_genomes.ndim == 2, \
        'target_genomes must be one-dimensional or two-dimensional.'
    if target_genomes.ndim == 1:
        assert pool_frequencies.shape == target_genomes.shape, \
            'pool_frequencies and target_genome must have the same shape.'
    else:
        assert pool_frequencies.shape[0] == target_genomes.shape[1], \
            'The first dimension of pool_frequencies must be the same as the second dimension of target_genomes.'
    pool_frequencies = np.clip(pool_frequencies, epsilon, 1 - epsilon)
    reference_frequencies = np.clip(reference_frequencies, epsilon, 1 - epsilon)
    alternate_frequency_ratios = pool_frequencies / reference_frequencies
    reference_frequency_ratios = (1 - pool_frequencies) / (1 - reference_frequencies)
    alternate_terms = target_genomes * np.log(alternate_frequency_ratios)
    reference_terms = (1 - target_genomes) * np.log(reference_frequency_ratios)
    return np.sum(alternate_terms + reference_terms, axis=1)