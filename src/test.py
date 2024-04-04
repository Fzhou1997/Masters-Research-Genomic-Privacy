import torch
import torchmetrics
from torch.nn.functional import one_hot
from torchmetrics import Metric


def analyze_metric(metric: Metric) -> None:
    # Analyze the metric's state
    metric.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    print(f"Metric score: {metric.compute()}")


if __name__ == "__main__":
    # Example usage with a specific metric
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
    analyze_metric(accuracy_metric)
