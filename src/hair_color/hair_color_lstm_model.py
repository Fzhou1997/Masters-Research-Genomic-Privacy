import os

import torch
import torch.nn as nn

class HairColorLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HairColorLSTMModel, self).__init__()
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

        pass

    def eval(self, test_loader, criterion):
        pass

    def save(self, file_path, file_name):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(file_path, file_name))

    def load(self, file_path, file_name):
        self.load_state_dict(torch.load(os.path.join(file_path, file_name)))
