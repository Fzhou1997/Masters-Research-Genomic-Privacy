import torch
import torch.nn as nn
from torch import Tensor

from .ModelAttackerLSTM import ModelAttackerLSTM


class ModelAttackerLSTMLinear(ModelAttackerLSTM):

    linear: nn.Linear

    def __init__(self,
                 lstm_input_size: int,
                 lstm_hidden_size: int,
                 lstm_num_layers: int = 1,
                 lstm_bidirectional: bool = False,
                 lstm_dropout: float = 0.5,
                 linear_output_size: int = 1):
        super(ModelAttackerLSTMLinear, self).__init__(lstm_input_size=lstm_input_size,
                                                      lstm_hidden_size=lstm_hidden_size,
                                                      lstm_num_layers=lstm_num_layers,
                                                      lstm_bidirectional=lstm_bidirectional,
                                                      lstm_dropout=lstm_dropout)
        self.linear = nn.Linear(in_features=lstm_hidden_size * (2 if lstm_bidirectional else 1),
                                out_features=linear_output_size)

    @property
    def linear_output_size(self) -> int:
        return self.linear.out_features

    def forward(self,
                x: Tensor,
                hidden: Tensor,
                cell: Tensor) -> tuple[tuple[Tensor, Tensor], Tensor]:
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        if self.lstm_bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            last_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            last_hidden = hidden[-1, :, :]
        logits = self.linear(last_hidden).squeeze()
        return (hidden, cell), logits

    def predict(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits)

    def classify(self, predicted: Tensor) -> Tensor:
        return torch.round(predicted)


