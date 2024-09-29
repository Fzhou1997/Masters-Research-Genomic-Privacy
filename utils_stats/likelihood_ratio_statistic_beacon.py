import math

import numpy as np
import numpy.typing as npt
import scipy.special as sp


def likelihood_ratio_statistic_beacon(
        target_genomes: npt.NDArray[np.bool_],
        beacon_responses: npt.NDArray[np.bool_],
        alpha_0: float,
        beta_0: float,
        num_beacon_genomes: int,
        probability_mismatch: float = 1e-6,
        epsilon: float = 1e-8) -> float | npt.NDArray[np.float64]:
    """
    Calculate the likelihood ratio statistic for a set of beacon responses given the target genomes,
    prior parameters, and mismatch probability.

    Args:
        target_genomes (npt.NDArray[np.bool_]): An array of boolean values representing the target genomes.
        beacon_responses (npt.NDArray[np.bool_]): An array of boolean values representing the beacon responses.
        alpha_0 (float): The alpha parameter of the prior distribution.
        beta_0 (float): The beta parameter of the prior distribution.
        num_beacon_genomes (int): The number of genomes in the beacon.
        probability_mismatch (float, optional): The mismatch probability. Defaults to 1e-6.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float | npt.NDArray[np.float64]: The likelihood ratio statistic of the responses.

    Raises:
        AssertionError: If the dimensions of the input arrays do not match the expected shapes.
        AssertionError: If the probability of mismatch is not between 0 and 1.
    """
    assert beacon_responses.ndim == 1, \
        "The beacon responses must be a 1D array."
    assert target_genomes.ndim == 1 or target_genomes.ndim == 2, \
        "The target genomes must be a 1D or 2D array."
    if target_genomes.ndim == 1:
        assert target_genomes.shape[0] == beacon_responses.shape[0], \
            "The number of target genomes must be the same as the number of beacon responses."
    else:
        assert target_genomes.shape[1] == beacon_responses.shape[0], \
            "The number of target genomes must be the same as the number of beacon responses."
    assert 0 <= probability_mismatch <= 1, \
        "The mismatch probability must be between 0 and 1."
    numerator = sp.gamma(alpha_0 + beta_0)
    denominator_absent = sp.gamma(beta_0) * math.pow((2 * num_beacon_genomes + alpha_0 + beta_0), alpha_0)
    denominator_unique = sp.gamma(beta_0) * math.pow((2 * (num_beacon_genomes - 1) + alpha_0 + beta_0), alpha_0)
    probability_absent = np.clip(numerator / denominator_absent, epsilon, 1 - epsilon)
    probability_unique = np.clip(numerator / denominator_unique, epsilon, 1 - epsilon)
    heterozygous_responses = target_genomes * beacon_responses
    log_present_null = math.log(1 - probability_absent)
    log_absent_null = math.log(probability_absent)
    log_present_alternative = math.log(1 - probability_mismatch * probability_unique)
    log_absent_alternative = math.log(probability_mismatch * probability_unique)
    axis = 1 if target_genomes.ndim == 2 else None
    log_likelihood_null = np.sum(heterozygous_responses * log_present_null + (1 - heterozygous_responses) * log_absent_null, axis=axis)
    log_likelihood_alternative = np.sum(heterozygous_responses * log_present_alternative + (1 - heterozygous_responses) * log_absent_alternative, axis=axis)
    return log_likelihood_null - log_likelihood_alternative


def likelihood_ratio_statistic_beacon_linearized(
        target_genomes: npt.NDArray[np.bool_],
        beacon_responses: npt.NDArray[np.bool_],
        alpha_0: float,
        beta_0: float,
        num_beacon_genomes: int,
        probability_mismatch: float = 1e-6,
        epsilon: float = 1e-8) -> float | npt.NDArray[np.float64]:
    """
    Calculate the likelihood ratio statistic for a set of beacon responses given the target genomes,
    prior parameters, and mismatch probability using a linearized approximation.

    Args:
        target_genomes (npt.NDArray[np.bool_]): An array of boolean values representing the target genomes.
        beacon_responses (npt.NDArray[np.bool_]): An array of boolean values representing the beacon responses.
        alpha_0 (float): The alpha parameter of the prior distribution.
        beta_0 (float): The beta parameter of the prior distribution.
        num_beacon_genomes (int): The number of genomes in the beacon.
        probability_mismatch (float, optional): The sequencing mismatch probability. Defaults to 1e-6.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float | npt.NDArray[np.float64]: The likelihood ratio statistic of the responses.

    Raises:
        AssertionError: If the dimensions of the input arrays do not match the expected shapes.
        AssertionError: If the probability of mismatch is not between 0 and 1.
    """
    assert beacon_responses.ndim == 1, \
        "The beacon responses must be a 1D array."
    assert target_genomes.ndim == 1 or target_genomes.ndim == 2, \
        "The target genomes must be a 1D or 2D array."
    if target_genomes.ndim == 1:
        assert target_genomes.shape[0] == beacon_responses.shape[0], \
            "The number of target genomes must be the same as the number of beacon responses."
    else:
        assert target_genomes.shape[1] == beacon_responses.shape[0], \
            "The number of target genomes must be the same as the number of beacon responses."
    assert 0 <= probability_mismatch <= 1, \
        "The mismatch probability must be between 0 and 1."
    numerator = sp.gamma(alpha_0 + beta_0)
    denominator_absent = sp.gamma(beta_0) * math.pow((2 * num_beacon_genomes + alpha_0 + beta_0), alpha_0)
    denominator_unique = sp.gamma(beta_0) * math.pow((2 * (num_beacon_genomes - 1) + alpha_0 + beta_0), alpha_0)
    probability_absent = np.clip(numerator / denominator_absent, epsilon, 1 - epsilon)
    probability_unique = np.clip(numerator / denominator_unique, epsilon, 1 - epsilon)
    b = math.log(probability_absent
                 / (probability_mismatch * probability_unique))
    c = math.log((probability_mismatch * probability_unique * (1 - probability_absent))
                 / (probability_absent * (1 - probability_mismatch * probability_unique)))
    axis = 1 if target_genomes.ndim == 2 else None
    num_heterozygous = np.sum(target_genomes, axis=axis)
    num_matches = np.sum(target_genomes * beacon_responses, axis=axis)
    return num_heterozygous * b + c * num_matches


def likelihood_ratio_statistic_beacon_optimized(
        target_genomes: npt.NDArray[np.bool_],
        beacon_presences: npt.NDArray[np.bool_],
        reference_frequencies: npt.NDArray[np.float64],
        num_beacon_genomes: int,
        probability_mismatch: float = 1e-6,
        epsilon: float = 1e-8) -> float | npt.NDArray[np.float64]:
    """
    Calculate the optimized membership likelihood ratio statistics for a set of beacon responses,
    given the target genomes, allele frequencies, and mismatch probability.

    Args:
        target_genomes (npt.NDArray[np.bool_]): An array of boolean values representing the target genomes.
        beacon_presences (npt.NDArray[np.bool_]): An array of boolean values representing the beacon responses.
        reference_frequencies (npt.NDArray[np.float64]): An array of float values representing the allele frequencies.
        num_beacon_genomes (int): The number of genomes in the beacon.
        probability_mismatch (float, optional): The sequencing mismatch probability. Defaults to 1e-6.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float | npt.NDArray[np.float64]: The optimized likelihood ratio statistic of the responses.

    Raises:
        AssertionError: If the dimensions of the input arrays do not match the expected shapes.
        AssertionError: If the probability of mismatch is not between 0 and 1.
    """
    assert beacon_presences.ndim == 1, \
        "The beacon responses must be a 1D array."
    assert reference_frequencies.ndim == 1, \
        "The allele frequencies must be a 1D array."
    assert target_genomes.ndim == 1 or target_genomes.ndim == 2, \
        "The target genomes must be a 1D or 2D array."
    assert beacon_presences.shape == reference_frequencies.shape, \
        "The beacon responses and allele frequencies must have the same shape."
    if target_genomes.ndim == 1:
        assert target_genomes.shape[0] == reference_frequencies.shape[0], \
            "The number of target genomes must be the same as the number of allele frequencies."
    else:
        assert target_genomes.shape[1] == reference_frequencies.shape[0], \
            "The number of target genomes must be the same as the number of allele frequencies."
    assert 0 <= probability_mismatch <= 1, \
        "The mismatch probability must be between 0 and 1."
    reference_frequencies = np.clip(reference_frequencies, epsilon, 1 - epsilon, dtype=np.longdouble)

    probabilities_square = np.clip(np.power(1 - reference_frequencies, 2, dtype=np.longdouble), epsilon, 1 - epsilon)
    probabilities_unique = np.clip(np.power(1 - reference_frequencies, 2 * num_beacon_genomes - 2, dtype=np.longdouble), epsilon, 1 - epsilon)
    probabilities_absent = np.clip(np.power(1 - reference_frequencies, 2 * num_beacon_genomes, dtype=np.longdouble), epsilon, 1 - epsilon)

    alternate_terms = (target_genomes * beacon_presences) * np.log((1 - probabilities_absent) / (1 - probability_mismatch * probabilities_unique))
    null_terms = (target_genomes * (1 - beacon_presences)) * np.log(probabilities_square / probability_mismatch)

    # log_likelihoods_null = np.log(probabilities_square / probability_mismatch, dtype=np.longdouble)
    # log_likelihoods_alternative = np.log((probability_mismatch / probabilities_square)
    #     * ((1 - probabilities_absent) / (1 - probability_mismatch * probabilities_unique)), dtype=np.longdouble)

    axis = 1 if target_genomes.ndim == 2 else None
    return np.sum(alternate_terms + null_terms, axis=axis)