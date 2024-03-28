import os

import torch
import torch.nn as nn

from torchmetrics import Accuracy, Precision, Recall, F1Score


class HairColorLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HairColorLSTMModel, self).__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

    def train(self, train_loader, criterion, optimizer, num_epochs):
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

    def eval(self, test_loader, criterion):
        self.training = False
        eval_loss = 0.0
        accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)
        precision = Precision(task='multiclass', num_classes=self.num_classes)
        recall = Recall(task='multiclass', num_classes=self.num_classes)
        f1 = F1Score(task='multiclass', num_classes=self.num_classes)
        with torch.no_grad():
            for genotype, phenotype in test_loader:
                predicted = self(genotype)
                eval_loss += criterion(predicted, phenotype).item()
                accuracy.update(predicted, phenotype)
                precision.update(predicted, phenotype)
                recall.update(predicted, phenotype)
                f1.update(predicted, phenotype)
        eval_loss /= len(test_loader)
        return eval_loss, accuracy.compute(), precision.compute(), recall.compute(), f1.compute()

    def save(self, file_path, file_name):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(file_path, file_name))

    def load(self, file_path, file_name):
        self.load_state_dict(torch.load(os.path.join(file_path, file_name)))
