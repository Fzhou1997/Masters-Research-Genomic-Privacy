import math


def log_likelihood_response(
        responses: list[bool],
        probabilities_true: list[float],
        probabilities_false: list[float],
        epsilon: float = 1e-8) -> float:
    assert len(responses) == len(probabilities_true) == len(probabilities_false)
    out = 0
    for i in range(len(responses)):
        response = int(responses[i])
        probability_true = max(probabilities_true[i], epsilon)
        probability_false = max(probabilities_false[i], epsilon)
        out += response * math.log(probability_true) + (1 - response) * math.log(probability_false)
    return out


def log_likelihood_alternative(
        responses: list[bool],
        probabilities_unique: list[float],
        probability_mismatch: float,
        epsilon: float = 1e-8) -> float:
    assert len(responses) == len(probabilities_unique)
    out = 0
    for i in range(len(responses)):
        response = int(responses[i])
        probability_unique = max(probabilities_unique[i], epsilon)
        log_present = math.log(1 - probability_mismatch * probability_unique)
        log_absent = math.log(probability_mismatch * probability_unique)
        out += response * log_present + (1 - response) * log_absent
    return out


def log_likelihood_null(
        responses: list[bool],
        probabilities_absent: list[float],
        epsilon: float = 1e-8) -> float:
    assert len(responses) == len(probabilities_absent)
    out = 0
    for i in range(len(responses)):
        response = int(responses[i])
        probability_absent = max(probabilities_absent[i], epsilon)
        log_present = math.log(1 - probability_absent)
        log_absent = math.log(probability_absent)
        out += response * log_present + (1 - response) * log_absent
    return out


def likelihood_ratio_statistic(
        responses: list[bool],
        probabilities_unique: list[float],
        probabilities_absent: list[float],
        probability_mismatch: float,
        epsilon: float = 1e-8) -> float:
    assert len(responses) == len(probabilities_unique) == len(probabilities_absent)
    null = log_likelihood_null(responses, probabilities_absent, epsilon)
    alternative = log_likelihood_alternative(responses, probabilities_unique, probability_mismatch, epsilon)
    return null - alternative
