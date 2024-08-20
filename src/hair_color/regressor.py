import os
from typing import Self

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix

from src.hair_color.dataloader import HairColorDataLoader

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Regressor(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool,
                 dropout: float):
        super(Regressor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hidden, cell) = self.lstm(x)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        y = self.linear(hidden)
        return y

    def save(self, file_path: str | bytes | os.PathLike[str] | os.PathLike[bytes], file_name: str) -> None:
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.state_dict(), f'{file_path}/{file_name}.pth')

    def load(self, file_path: str | bytes | os.PathLike[str] | os.PathLike[bytes], file_name: str) -> Self:
        self.load_state_dict(torch.load(f'{file_path}/{file_name}.pth'))
        return self


def classify(predicted: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(predicted), 0, 2).long()


class Trainer:
    def __init__(self,
                 model: Regressor,
                 loss: _Loss,
                 optimizer: Optimizer,
                 scheduler: LRScheduler,
                 train_loader: HairColorDataLoader,
                 test_loader: HairColorDataLoader,
                 out_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                 out_name: str):
        self.model = model
        self.loss = loss
        self.accuracy = Accuracy(task='multiclass', num_classes=3)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.out_path = out_path
        self.out_name = out_name

    def _train(self) -> tuple[float, float]:
        running_loss = 0
        self.accuracy.reset()
        self.model.train()
        for features, labels in self.train_loader:
            self.optimizer.zero_grad()
            predicted = self.model(features)
            loss = self.loss(predicted, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()
            pred = classify(predicted)
            self.accuracy.update(pred, labels)
        running_loss /= len(self.train_loader)
        accuracy = self.accuracy.compute().cpu().item()
        return running_loss, accuracy

    def _validate(self) -> tuple[float, float]:
        loss = 0
        self.accuracy.reset()
        self.model.eval()
        with torch.no_grad():
            for features, labels in self.test_loader:
                predicted = self.model(features)
                loss += self.loss(predicted, labels).item()
                pred = classify(predicted)
                self.accuracy.update(pred, labels)
        loss /= len(self.test_loader)
        accuracy = self.accuracy.compute().cpu().item()
        return loss, accuracy

    def train(self, num_epochs: int) -> tuple[list[float], list[float], list[float], list[float]]:
        best_loss = float('inf')
        training_losses = []
        training_accuracies = []
        validation_losses = []
        validation_accuracies = []
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self._train()
            validation_loss, validation_accuracy = self._validate()
            training_losses.append(train_loss)
            training_accuracies.append(train_accuracy)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Training Loss: {train_loss}, Training Accuracy: {train_accuracy}')
            print(f'Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')
            if validation_loss < best_loss:
                print(f'Validation Loss Improved: {best_loss} -> {validation_loss}. Saving Model...')
                best_loss = validation_loss
                self.model.save(self.out_path, self.out_name)
            print(25 * "==")
        return training_losses, training_accuracies, validation_losses, validation_accuracies


class Tester:
    def __init__(self,
                 model: Regressor,
                 loss: _Loss,
                 test_loader: HairColorDataLoader):
        self.model = model
        self.loss = loss
        self.accuracy = Accuracy(task='multiclass', num_classes=3)
        self.f1_score = F1Score(task='multiclass', num_classes=3)
        self.precision = Precision(task='multiclass', num_classes=3)
        self.recall = Recall(task='multiclass', num_classes=3)
        self.auroc = AUROC(task='multiclass', num_classes=3)
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=3)
        self.test_loader = test_loader

    def test(self) -> tuple[float, float, float, float, float, float, list[list[int]]]:
        self.model.eval()
        loss = 0
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        self.auroc.reset()
        self.confusion_matrix.reset()
        with torch.no_grad():
            for features, labels in self.test_loader:
                predicted = self.model(features)
                loss += self.loss(predicted, labels).item()
                pred = classify(predicted)
                self.accuracy.update(pred, labels)
                self.precision.update(pred, labels)
                self.recall.update(pred, labels)
                self.f1_score.update(pred, labels)
                self.auroc.update(pred, labels)
                self.confusion_matrix.update(pred, labels)
        loss /= len(self.test_loader)
        accuracy = self.accuracy.compute().cpu().item()
        precision = self.precision.compute().cpu().item()
        recall = self.recall.compute().cpu().item()
        f1_score = self.f1_score.compute().cpu().item()
        auroc = self.auroc.compute().cpu().item()
        cm = self.confusion_matrix.compute().cpu().tolist()
        return loss, accuracy, precision, recall, f1_score, auroc, cm
