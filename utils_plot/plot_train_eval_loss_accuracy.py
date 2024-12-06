import os
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_train_eval_loss_accuracy(train_loss: list[float] | npt.NDArray[np.float_],
                                  train_accuracy: list[float] | npt.NDArray[np.float_],
                                  eval_loss: list[float] | npt.NDArray[np.float_],
                                  eval_accuracy: list[float] | npt.NDArray[np.float_],
                                  saved_epoch: int = None,
                                  output_path: str | bytes | PathLike[str] | PathLike[bytes] = None,
                                  output_file: str = None,
                                  show: bool = True) -> None:
    """
    Plot the training and evaluation loss and accuracy.

    Args:
        train_loss (list[float] | npt.NDArray[np.float_]): list of training loss per epoch
        train_accuracy (list[float] | npt.NDArray[np.float_]): list of training per epoch
        eval_loss (list[float] | npt.NDArray[np.float_]): list of evaluation loss per epoch
        eval_accuracy (list[float] | npt.NDArray[np.float_]): list of evaluation accuracy per epoch
        saved_epoch (int, optional): epoch to save the plot. Defaults to None.
        output_path (str | bytes | PathLike[str] | PathLike[bytes], optional): path to save the plot. Defaults to None.
        output_file (str, optional): name of the output file. Defaults to None.
        show (bool, optional): display the plot. Defaults to True.
    """
    assert len(train_loss) == len(train_accuracy) == len(eval_loss) == len(eval_accuracy)
    epochs = list(range(1, len(train_loss) + 1))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    ax1.plot(epochs, train_loss, label="Train Loss", color="tab:red")
    ax1.plot(epochs, eval_loss, label="Eval Loss", color="tab:blue")
    if saved_epoch is not None:
        ax1.axvline(x=saved_epoch, color="tab:grey", linestyle="--", label="Saved Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y')
    ax1.set_title("Losses by Epoch")
    ax1.legend(loc="upper right")

    ax2.plot(epochs, train_accuracy, label="Train Accuracy", color="tab:red")
    ax2.plot(epochs, eval_accuracy, label="Eval Accuracy", color="tab:blue")
    if saved_epoch is not None:
        ax2.axvline(x=saved_epoch, color="tab:grey", linestyle="--", label="Saved Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.tick_params(axis='y')
    ax2.set_title("Accuracy by Epoch")
    ax2.legend(loc="upper right")

    if output_file is not None:
        if output_path is None:
            output_path = os.getcwd()
        plt.savefig(os.path.join(output_path, output_file))

    if show:
        plt.show()
