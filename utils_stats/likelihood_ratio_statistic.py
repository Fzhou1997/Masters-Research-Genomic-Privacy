from utils_stats.log_likelihood_null import log_likelihood_null
from utils_stats.log_likelihood_alternative import log_likelihood_alternative


def likelihood_ratio_statistic(
        responses: list[bool],
        probabilities_unique: list[float],
        probabilities_absent: list[float],
        probability_mismatch: float,
        epsilon: float = 1e-8) -> float:
    """
    Calculate the membership likelihood ratio statistic for a set of responses given the unique and absent probabilities.

    Args:
        responses (list[bool]): A list of boolean responses.
        probabilities_unique (list[float]): The probabilities of the snps being unique in the beacon.
        probabilities_absent (list[float]): The probabilities of the snps being absent in the beacon.
        probability_mismatch (float): The mismatch probability.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float: The likelihood ratio statistic of the responses.

    Raises:
        AssertionError: If the lengths of the input lists are not equal.
    """
    assert len(responses) == len(probabilities_unique) == len(probabilities_absent)
    null = log_likelihood_null(responses, probabilities_absent, epsilon)
    alternative = log_likelihood_alternative(responses, probabilities_unique, probability_mismatch, epsilon)
    return null - alternative
