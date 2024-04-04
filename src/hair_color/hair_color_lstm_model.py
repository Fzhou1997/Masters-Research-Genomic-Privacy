import os

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from torchmetrics import Accuracy, Precision, Recall, F1Score, Metric

from src.hair_color.hair_color_dataloader import HairColorDataLoader


class HairColorLSTMModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_label_classes: int):
        super(HairColorLSTMModel, self).__init__()
        self.num_label_classes = num_label_classes
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, num_label_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        predicted, _ = self.lstm(data)
        predicted = self.linear(predicted[:, -1, :])
        return predicted

    def train(self,
              train_loader: HairColorDataLoader,
              criterion: _Loss,
              optimizer: Optimizer,
              num_epochs: int) -> list[float]:
        self.training = True
        training_losses = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for genotype, phenotype in train_loader:
                outputs = self(genotype)
                optimizer.zero_grad()
                loss = criterion(outputs, phenotype)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            training_losses.append(epoch_loss)
        return training_losses

    def eval(self,
             test_loader: HairColorDataLoader,
             criterion: _Loss,
             metrics: list[Metric]) -> tuple[float, float, float, float, float]:
        self.training = False
        eval_loss = 0.0
        accuracy = Accuracy(task='multiclass', num_classes=self.num_label_classes)
        precision = Precision(task='multiclass', num_classes=self.num_label_classes)
        recall = Recall(task='multiclass', num_classes=self.num_label_classes)
        f1 = F1Score(task='multiclass', num_classes=self.num_label_classes)
        with torch.no_grad():
            for genotype, phenotype in test_loader:
                predicted = self(genotype)
                eval_loss += criterion(predicted, phenotype).item()
                if test_loader.is_labels_one_hot():
                    phenotype = phenotype.argmax(dim=1)
                    predicted = predicted.argmax(dim=1)
                accuracy.update(predicted, phenotype)
                precision.update(predicted, phenotype)
                recall.update(predicted, phenotype)
                f1.update(predicted, phenotype)
        eval_loss /= len(test_loader)
        return eval_loss, accuracy.compute(), precision.compute(), recall.compute(), f1.compute()

    def predict(self, data):
        self.training = False
        with torch.no_grad():
            predicted = self(data)
            return predicted

    def save(self,
             file_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
             file_name):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(file_path, file_name))

    def load(self, file_path, file_name):
        self.load_state_dict(torch.load(os.path.join(file_path, file_name)))
