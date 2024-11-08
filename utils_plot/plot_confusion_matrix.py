import os
from os import PathLike
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_confusion_matrix(confusion_matrix: list[list[int]] | npt.NDArray[np.int_],
                          task: Literal["binary", "multiclass"],
                          output_path: str | bytes | PathLike[str] | PathLike[bytes] = None,
                          output_file: str = None) -> None:
    """
    Plot the confusion matrix.

    Args:
        confusion_matrix (list[list[int] | npt.NDArray[np.int_]]): confusion matrix
        task (Literal["binary", "multiclass"]): task type
        output_path (str | bytes | PathLike[str] | PathLike[bytes], optional): path to save the plot. Defaults to None.
        output_file (str, optional): name of the output file. Defaults to None.
    """
    if isinstance(confusion_matrix, list):
        num_classes = len(confusion_matrix)
        for row in confusion_matrix:
            assert len(row) == num_classes
    else:
        num_classes = confusion_matrix.shape[0]
        for row in confusion_matrix:
            assert row.shape[0] == num_classes

    if task == "binary":
        if isinstance(confusion_matrix, list):
            confusion_matrix = np.array([[confusion_matrix[1][1], confusion_matrix[1][0]],
                                         [confusion_matrix[0][1], confusion_matrix[0][0]]])
        else:
            confusion_matrix = np.array([[confusion_matrix[1, 1], confusion_matrix[1, 0]],
                                         [confusion_matrix[0, 1], confusion_matrix[0, 0]]])


    fig, ax = plt.subplots(figsize=(8, 6))
    vmin = 0
    vmax = np.max(confusion_matrix)

    mask = np.eye(num_classes, dtype=bool)
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Reds",
                cbar=False,
                annot_kws={"size": 16},
                linewidths=0.5,
                ax=ax,
                mask=mask,
                square=True,
                linecolor="white",
                vmin=vmin,
                vmax=vmax)

    inverse_mask = ~mask
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                annot_kws={"size": 16},
                linewidths=0.5,
                ax=ax,
                mask=inverse_mask,
                square=True,
                linecolor="white",
                vmin=vmin,
                vmax=vmax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrix", y=0.05)
    plt.gca().xaxis.set_ticks_position("top")
    plt.gca().xaxis.set_label_position("top")
    ax.set_xticklabels(["Positive", "Negative"], rotation=0)
    ax.set_yticklabels(["Positive", "Negative"], rotation=0)

    if output_file is not None:
        if output_path is None:
            output_path = os.getcwd()
        plt.savefig(os.path.join(output_path, output_file))

    plt.show()
