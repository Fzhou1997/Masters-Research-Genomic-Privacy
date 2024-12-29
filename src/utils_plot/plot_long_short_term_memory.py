import os

from matplotlib import pyplot as plt
from torch import Tensor


def plot_long_short_term_memory(long_term_memory: Tensor,
                                short_term_memory: Tensor,
                                bidirectional: bool = False,
                                xlabel: str = "Time",
                                ylabel: str = "Memory",
                                title: str = "Long and Short Term Memory",
                                output_path: str | bytes | os.PathLike[str] | os.PathLike[bytes] = None,
                                output_file: str = None,
                                show: bool = True) -> None:
    assert long_term_memory.dim() == 2 and short_term_memory.dim() == 2, \
        "The long term memory and short term memory must not be batched."
    assert long_term_memory.shape[0] == short_term_memory.shape[0], \
        "The number of long term memory and short term memory must have the same sequence length."

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    hidden_size = short_term_memory.shape[1]
    for i in range(hidden_size):
        is_reverse = (i >= (hidden_size // 2)) if bidirectional else False
        color = "tab:red" if is_reverse else "tab:blue"
        ax1.plot(short_term_memory[:, i], color=color, alpha=4 / hidden_size)
    if bidirectional:
        hidden_forward_mean = short_term_memory[:, :hidden_size // 2].mean(dim=1)
        hidden_reverse_mean = short_term_memory[:, hidden_size // 2:].mean(dim=1)
        ax1.plot(hidden_forward_mean, color="tab:blue", alpha=1, label="Short Term Memory (Forward)")
        ax1.plot(hidden_reverse_mean, color="tab:red", alpha=1, label="Short Term Memory (Reverse)")
    else:
        hidden_mean = short_term_memory.mean(dim=1)
        ax1.plot(hidden_mean, color="tab:green", alpha=1, label="Short Term Memory")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend(loc="upper right")

    cell_size = long_term_memory.shape[1]
    for i in range(cell_size):
        is_reverse = (i >= (cell_size // 2)) if bidirectional else False
        color = "tab:red" if is_reverse else "tab:blue"
        ax2.plot(long_term_memory[:, i], color=color, alpha=4 / cell_size)
    if bidirectional:
        cell_forward_mean = long_term_memory[:, :cell_size // 2].mean(dim=1)
        cell_reverse_mean = long_term_memory[:, cell_size // 2:].mean(dim=1)
        ax2.plot(cell_forward_mean, color="tab:blue", alpha=1, label="Long Term Memory (Forward)")
        ax2.plot(cell_reverse_mean, color="tab:red", alpha=1, label="Long Term Memory (Reverse)")
    else:
        cell_mean = long_term_memory.mean(dim=1)
        ax2.plot(cell_mean, color="tab:green", alpha=1, label="Long Term Memory")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.legend(loc="upper right")

    fig.suptitle(title)

    if output_file is not None:
        if output_path is None:
            output_path = os.getcwd()
        plt.savefig(os.path.join(output_path, output_file))

    if show:
        plt.show()
    else:
        plt.close()
