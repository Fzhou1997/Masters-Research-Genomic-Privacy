import numpy as np
import numpy.typing as npt


def likelihood_ratio_statistic_frequency(
        pool_frequencies: npt.NDArray[np.float64],
        population_frequencies: npt.NDArray[np.float64],
        alternate_allele_counts: npt.NDArray[np.int64],
        epsilon: float = 1e-8) -> float:
    return np.sum(

    )