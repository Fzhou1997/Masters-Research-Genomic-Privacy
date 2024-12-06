from os import PathLike

import matplotlib.pyplot as plt


def plot_likelihood_ratio_test_hyper_num_snps(num_snps: list[int],
                                              acc: list[float],
                                              prec: list[float],
                                              rec: list[float],
                                              f1: list[float],
                                              roc_auc: list[float],
                                              width: float = 6,
                                              height: float = 6,
                                              title: str = "LRT Metrics By Sequence Length",
                                              xlabel: str = "Sequence Length",
                                              ylabel: str = "Metric Values",
                                              output_file: str | bytes | PathLike[str] | PathLike[bytes] = None,
                                              show: bool = True) -> None:
    assert len(num_snps) == len(acc) == len(prec) == len(rec) == len(f1) == len(roc_auc), \
        "The number of SNPs, accuracy, precision, recall, F1, and ROC AUC must be the same."
    plt.figure(figsize=(width, height))
    plt.plot(num_snps, acc, color="blue", label="Accuracy")
    plt.plot(num_snps, prec, color="green", label="Precision")
    plt.plot(num_snps, rec, color="red", label="Recall")
    plt.plot(num_snps, f1, color="purple", label="F1")
    plt.plot(num_snps, roc_auc, color="orange", label="AUROC")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="x")
    if output_file is not None:
        plt.savefig(output_file)
    if show:
        plt.show()
    else:
        plt.close()

