import math


def log_likelihood_alternative(
        responses: list[bool],
        probabilities_unique: list[float],
        probability_mismatch: float,
        epsilon: float = 1e-8) -> float:
    """
    Calculate the log-likelihood of the alternative hypothesis (the given genome is present in the beacon) of
    a set of beacon responses given the probabilities of snps being unique in the beacon and the mismatch probability.

    Args:
        responses (list[bool]): A list of boolean responses.
        probabilities_unique (list[float]): The probabilities of the snps being unique in the beacon.
        probability_mismatch (float): The mismatch probability.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float: The log-likelihood of the alternative hypothesis (the given genome is present in the beacon) of a
        set of beacon responses.

    Raises:
        AssertionError: If the lengths of the input lists are not equal.
    """
    assert len(responses) == len(probabilities_unique)
    out = 0
    for i in range(len(responses)):
        response = int(responses[i])
        probability_unique = max(probabilities_unique[i], epsilon)
        log_present = math.log(1 - probability_mismatch * probability_unique)
        log_absent = math.log(probability_mismatch * probability_unique)
        out += response * log_present + (1 - response) * log_absent
    return out
