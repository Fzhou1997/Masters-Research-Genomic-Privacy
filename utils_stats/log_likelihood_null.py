import math


def log_likelihood_null(
        responses: list[bool],
        probabilities_absent: list[float],
        epsilon: float = 1e-8) -> float:
    """
    Calculate the log-likelihood of the null hypothesis (the given genome is absent in the beacon) of a set of beacon
    responses given the probabilities of snps being absent in the beacon.

    Args:
        responses (list[bool]): A list of boolean responses.
        probabilities_absent (list[float]): The probabilities of the snps being absent in the beacon.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float: The log-likelihood of the null hypothesis (the given genome is absent in the beacon) of a set of beacon
        responses.

    Raises:
        AssertionError: If the lengths of the input lists are not equal.
    """
    assert len(responses) == len(probabilities_absent)
    out = 0
    for i in range(len(responses)):
        response = int(responses[i])
        probability_absent = max(probabilities_absent[i], epsilon)
        log_present = math.log(1 - probability_absent)
        log_absent = math.log(probability_absent)
        out += response * log_present + (1 - response) * log_absent
    return out
