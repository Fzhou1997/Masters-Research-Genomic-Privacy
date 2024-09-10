import math


def log_likelihood_response(
        responses: list[bool],
        probabilities_true: list[float],
        probabilities_false: list[float],
        epsilon: float = 1e-8) -> float:
    """
    Calculate the log-likelihood of a set of beacon responses given the probabilities of true and false responses.

    Args:
        responses (list[bool]): A list of boolean beacon responses.
        probabilities_true (list[float]): A list of probabilities corresponding to the true responses.
        probabilities_false (list[float]): A list of probabilities corresponding to the false responses.
        epsilon (float, optional): A small value to avoid taking the log of zero. Defaults to 1e-8.

    Returns:
        float: The log-likelihood of the responses.

    Raises:
        AssertionError: If the lengths of the input lists are not equal.
    """
    assert len(responses) == len(probabilities_true) == len(probabilities_false)
    out = 0
    for i in range(len(responses)):
        response = int(responses[i])
        probability_true = max(probabilities_true[i], epsilon)
        probability_false = max(probabilities_false[i], epsilon)
        out += response * math.log(probability_true) + (1 - response) * math.log(probability_false)
    return out
