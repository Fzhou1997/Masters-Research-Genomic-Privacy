from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_accuracy_threshold_curve(
        thresholds: list[float] | npt.NDArray[np.float64],
        accuracies: list[float] | npt.NDArray[np.float64],
        title: str = "Accuracy vs. Threshold",
        xlabel: str = "Threshold",
        ylabel: str = "Accuracy",
        output_file: str | bytes | PathLike[str] | PathLike[bytes] = None) -> None:
    plt.plot(thresholds, accuracies)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
