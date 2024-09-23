import math

import numpy as np
import numpy.typing as npt
import scipy.special as sp


def probability_absent_beacon_approximate(
        alpha_0: float,
        beta_0: float,
        num_genomes: int) -> float:
    """
    Calculate the approximate probability of the absence of any given snp in a beacon.

    This function uses sterling's approximation to compute the approximate probability
    that any given snp is absent from a beacon
    given the parameters of the beta distribution and the number of genomes.

    Parameters:
    alpha_0 (float): The alpha parameter of the beta distribution.
    beta_0 (float): The beta parameter of the beta distribution.
    num_genomes (int): The number of genomes.

    Returns:
    float: The approximate probability of the absence of any given snp in a beacon.
    """
    numerator = sp.gamma(alpha_0 + beta_0)
    denominator = sp.gamma(beta_0) * math.pow((2 * num_genomes + alpha_0 + beta_0), alpha_0)
    return numerator / denominator


def probability_absent_beacon(
        alpha_0: float,
        beta_0: float,
        num_genomes: int) -> float:
    """
    Calculate the approximate probability of the absence of any given snp in a beacon.

    This function uses an exact formula to compute the approximate probability
    that any given snp is absent from a beacon
    given the parameters of the beta distribution and the number of genomes.

    Parameters:
    alpha_0 (float): The alpha parameter of the beta distribution.
    beta_0 (float): The beta parameter of the beta distribution.
    num_genomes (int): The number of genomes.

    Returns:
    float: The exact probability of the absence of a beacon.
    """
    product = 1
    for r in range(2 * num_genomes):
        product *= (beta_0 + r) / (alpha_0 + beta_0 + r)
    return product


def probability_unique_beacon_approximate(
        alpha_0: float,
        beta_0: float,
        num_genomes: int) -> float:
    """
    Calculate the approximate probability of the absence
    of any given snp in all genomes except one in the beacon.

    This function uses sterling's approximation to compute the approximate probability
    that any given snp is absent in all genomes except one in the beacon
    given the parameters of the beta distribution and the number of genomes.

    Parameters:
    alpha_0 (float): The alpha parameter of the beta distribution.
    beta_0 (float): The beta parameter of the beta distribution.
    num_genomes (int): The number of genomes.

    Returns:
    float: The approximate probability of the absence of any given snp in a beacon.
    """
    numerator = sp.gamma(alpha_0 + beta_0)
    denominator = sp.gamma(beta_0) * math.pow((2 * (num_genomes - 1) + alpha_0 + beta_0), alpha_0)
    return numerator / denominator


def probability_unique_beacon(
        alpha_0: float,
        beta_0: float,
        num_genomes: int) -> float:
    """
    Calculate the exact probability of the absence
    of any given snp in all genomes except one in the beacon.

    This function uses an exact formula to compute the exact probability
    that any given snp is absent in all genomes except one in the beacon
    given the parameters of the beta distribution and the number of genomes.

    Parameters:
    alpha_0 (float): The alpha parameter of the beta distribution.
    beta_0 (float): The beta parameter of the beta distribution.
    num_genomes (int): The number of genomes.

    Returns:
    float: The exact probability of the absence of a beacon.
    """
    product = 1
    for r in range(2 * (num_genomes - 1)):
        product *= (beta_0 + r) / (alpha_0 + beta_0 + r)
    return product


def log_likelihood_null(
        responses: npt.NDArray[np.bool_],
        probability_absent: float,
        epsilon: float = 1e-8) -> float:
    """
    Calculate the log-likelihood of the null hypothesis (the given genome is absent in the beacon)
    of a set of beacon responses given the probability of any given snps being absent in the beacon.

    Args:
        responses (npt.NDArray[np.bool_]): An array of boolean values representing the beacon responses.
        probability_absent (float): The probability of any given snp being absent in the beacon.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float: The log-likelihood of the null hypothesis (the given genome is absent in the beacon) of a set of beacon
        responses.

    Raises:
        AssertionError: If the probability of absence is not between 0 and 1.
    """
    assert 0 <= probability_absent <= 1
    probability_absent = max(probability_absent, epsilon)
    log_present = math.log(1 - probability_absent)
    log_absent = math.log(probability_absent)
    return np.sum(responses * log_present + (1 - responses) * log_absent)


def log_likelihood_alternative(
        responses: npt.NDArray[np.bool_],
        probability_unique: float,
        probability_mismatch: float,
        epsilon: float = 1e-8) -> float:
    """
    Calculate the log-likelihood of the alternative hypothesis (the given genome is present in the beacon) of
    a set of beacon responses given the probability of any given snps being absent in all genomes except one in the beacon
    and the mismatch probability.

    Args:
        responses (npt.NDArray[np.bool_]): An array of boolean values representing the beacon responses.
        probability_unique (float): The probability of any given snps being absent in all genomes except the one in the beacon.
        probability_mismatch (float): The mismatch probability.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float: The log-likelihood of the alternative hypothesis (the given genome is present in the beacon) of a
        set of beacon responses.

    Raises:
        AssertionError: If the probability of uniqueness is not between 0 and 1.
        AssertionError: If the probability of mismatch is not between 0 and 1.
    """
    assert 0 <= probability_unique <= 1
    assert 0 <= probability_mismatch <= 1
    probability_unique = max(probability_unique, epsilon)
    probability_mismatch = max(probability_mismatch, epsilon)
    log_present = math.log(1 - probability_mismatch * probability_unique)
    log_absent = math.log(probability_mismatch * probability_unique)
    return np.sum(responses * log_present + (1 - responses) * log_absent)


def likelihood_ratio_statistic_beacon(
        beacon_responses: npt.NDArray[np.bool_],
        probability_unique: float,
        probability_absent: float,
        probability_mismatch: float,
        epsilon: float = 1e-8) -> float:
    """
    Calculate the membership likelihood ratio statistic for a set of beacon responses,
    given the probabilities of absence, uniqueness, and mismatch of any given snps in the beacon.

    Args:
        beacon_responses (npt.NDArray[np.bool_]): An array of boolean values representing the beacon responses.
        probability_unique (float): The probability of any given snps being absent in all genomes except one in the beacon.
        probability_absent (float): The probability of any given snps being absent in the beacon.
        probability_mismatch (float): The mismatch probability.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float: The likelihood ratio statistic of the responses.

    Raises:
        AssertionError: If the probability of uniqueness is not between 0 and 1.
        AssertionError: If the probability of absence is not between 0 and 1.
        AssertionError: If the probability of mismatch is not between 0 and 1.
    """
    assert 0 <= probability_unique <= 1, "The unique probability must be between 0 and 1."
    assert 0 <= probability_absent <= 1, "The absent probability must be between 0 and 1."
    assert 0 <= probability_mismatch <= 1, "The mismatch probability must be between 0 and 1."
    null = log_likelihood_null(beacon_responses, probability_absent, epsilon)
    alternative = log_likelihood_alternative(beacon_responses, probability_unique, probability_mismatch, epsilon)
    return null - alternative


def likelihood_ratio_statistic_beacon_linearized(
        beacon_responses: npt.NDArray[np.bool_],
        probability_absent: float,
        probability_unique: float,
        probability_mismatch: float,
        epsilon: float = 1e-8) -> float:
    """
    Calculate the linearized membership likelihood ratio statistic for beacon responses.

    This function computes the likelihood ratio statistic for a set of beacon responses,
    given the probabilities of absence, uniqueness, and mismatch of any given snps in the beacon.

    Parameters:
        beacon_responses (npt.NDArray[np.bool_]): An array of boolean values representing the beacon responses.
        probability_absent (float): The probability that any given snp is absent in all genomes in the beacon
        probability_unique (float): The probability that any given snp is absent in all genomes except one in the beacon.
        probability_mismatch (float): The probability of a mismatch of any given snp in the beacon.
        epsilon (float, optional): A small value to prevent taking the log of zero. Default is 1e-8.

    Returns:
        float: The computed likelihood ratio statistic.

    Raises:
        AssertionError: If the probability of absence is not between 0 and 1.
        AssertionError: If the probability of uniqueness is not between 0 and 1.
        AssertionError: If the probability of mismatch is not between 0 and 1.
    """
    assert 0 <= probability_absent <= 1, "The absent probability must be between 0 and 1."
    assert 0 <= probability_unique <= 1, "The unique probability must be between 0 and 1."
    assert 0 <= probability_mismatch <= 1, "The mismatch probability must be between 0 and 1."
    n = len(beacon_responses)
    b = math.log(probability_absent
                 / (probability_mismatch * probability_unique)
                 + epsilon)
    c = math.log((probability_mismatch * probability_unique * (1 - probability_absent))
                 / (probability_absent * (1 - probability_mismatch * probability_unique))
                 + epsilon)
    return n * b + c * np.sum(beacon_responses)
