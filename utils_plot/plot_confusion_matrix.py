import os
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns


def plot_confusion_matrix(confusion_matrix: list[list[int]] | npt.NDArray[np.int_],
                          output_path: str | bytes | PathLike[str] | PathLike[bytes] = None,
                          output_file: str = None) -> None:
    """
    Plot the confusion matrix.

    Args:
        confusion_matrix (list[list[int] | npt.NDArray[np.int_]]): confusion matrix
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


    fig, ax = plt.subplots(figsize=(8, 6))

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
                linecolor="black",
                center=0.5)

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
                linecolor="black",
                center=0.5)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticklabels(range(num_classes))
    ax.set_yticklabels(range(num_classes), rotation=0)

    if output_file is not None:
        if output_path is None:
            output_path = os.getcwd()
        plt.savefig(os.path.join(output_path, output_file))

    plt.show()
