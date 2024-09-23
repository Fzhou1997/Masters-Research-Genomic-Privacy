from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_likelihood_ratio_statistics(
        likelihood_ratio_statistics: list[float] | npt.NDArray[np.float64],
        membership_labels: list[bool] | npt.NDArray[np.bool_],
        threshold: float = None,
        xlabel: str = "Index",
        ylabel: str = "Likelihood Ratio Statistic",
        title: str = "Scatter Plot of Likelihood Ratio Statistics",
        output_file: str | bytes | PathLike[str] | PathLike[bytes] = None) -> None:
    assert len(likelihood_ratio_statistics) == len(membership_labels), \
        "The number of likelihood ratio statistics and membership labels must be the same."
    member_statistics = likelihood_ratio_statistics[membership_labels]
    non_member_statistics = likelihood_ratio_statistics[~membership_labels]
    plt.figure(figsize=(10, 8))
    plt.scatter(range(len(member_statistics)), member_statistics, color="green", label="Member")
    plt.scatter(range(len(member_statistics), len(likelihood_ratio_statistics)), non_member_statistics, color="red", label="Non-Member")
    if threshold is not None:
        plt.axhline(y=threshold, color="blue", linestyle="-", label="Threshold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()

