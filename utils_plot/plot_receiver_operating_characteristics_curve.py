import os
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_receiver_operating_characteristics_curve(
        true_positive_rates: list[float] | npt.NDArray[np.float64],
        false_positive_rates: list[float] | npt.NDArray[np.float64],
        auc: float = None,
        xlabel: str = "False Positive Rate",
        ylabel: str = "True Positive Rate",
        title: str = "Receiver Operating Characteristics Curve",
        output_path: str | bytes | PathLike[str] | PathLike[bytes] = None,
        output_file: str = None,
        show: bool = True) -> None:
    assert len(true_positive_rates) == len(false_positive_rates), \
        "The number of true positive rates and false positive rates must be the same."

    plt.figure(figsize=(8, 8))
    plt.plot(false_positive_rates, true_positive_rates, color="blue", label=f"ROC Curve{'' if auc is None else f' (AUC = {auc:.4f})'}")
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if output_file is not None:
        if output_path is None:
            output_path = os.getcwd()
        plt.savefig(os.path.join(output_path, output_file))

    if show:
        plt.show()
