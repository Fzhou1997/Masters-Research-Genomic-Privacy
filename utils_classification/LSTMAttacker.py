import os
from typing import Self

import torch
import torch.nn as nn
from torch import Tensor


class LSTMAttacker(nn.Module):
    """
    LSTMAttacker is a neural network module that uses an LSTM layer followed by a linear layer
    for sequence classification tasks.

    Attributes:
        lstm (nn.LSTM): The LSTM layer.
        linear (nn.Linear): The linear layer for classification.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0.0,
                 output_size: int = 1):
        """
        Initializes the LSTMAttacker module.

        Args:
            input_size (int): The number of expected features in the input.
            hidden_size (int): The number of features in the hidden state.
            num_layers (int, optional): Number of recurrent layers. Default is 1.
            bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Default is False.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer. Default is 0.0.
            output_size (int, optional): The number of output features. Default is 1.
        """
        super(LSTMAttacker, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    @property
    def input_size(self) -> int:
        """Returns the input size of the LSTM."""
        return self.lstm.input_size

    @property
    def hidden_size(self) -> int:
        """Returns the hidden size of the LSTM."""
        return self.lstm.hidden_size

    @property
    def num_layers(self) -> int:
        """Returns the number of layers in the LSTM."""
        return self.lstm.num_layers

    @property
    def num_directions(self) -> int:
        """Returns the number of directions in the LSTM (1 for unidirectional, 2 for bidirectional)."""
        return 2 if self.lstm.bidirectional else 1

    @property
    def dropout(self) -> float:
        """Returns the dropout rate of the LSTM."""
        return self.lstm.dropout

    @property
    def is_bidirectional(self) -> bool:
        """Returns True if the LSTM is bidirectional, False otherwise."""
        return self.lstm.bidirectional

    def forward(self,
                x: Tensor,
                hidden: Tensor,
                cell: Tensor) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """
        Defines the forward pass of the LSTMAttacker.

        Args:
            x (Tensor): The input tensor.
            hidden (Tensor): The hidden state tensor.
            cell (Tensor): The cell state tensor.

        Returns:
            tuple: A tuple containing the new hidden and cell states, and the logits.
        """
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        if self.is_bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            hidden = hidden[-1, :, :]
        logits = self.linear(hidden).squeeze()
        return (hidden, cell), logits

    def init_hidden_cell(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """
        Initializes the hidden and cell states with zeros.

        Args:
            batch_size (int): The batch size.

        Returns:
            tuple: A tuple containing the initialized hidden and cell states.
        """
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        return hidden, cell

    def predict(self, logits: Tensor) -> Tensor:
        """
        Applies a sigmoid activation to the logits to get the prediction probabilities.

        Args:
            logits (Tensor): The logits tensor.

        Returns:
            Tensor: The prediction probabilities.
        """
        return torch.sigmoid(logits)

    def classify(self, predicted: Tensor) -> Tensor:
        """
        Rounds the prediction probabilities to get binary classification results.

        Args:
            predicted (Tensor): The prediction probabilities.

        Returns:
            Tensor: The binary classification results.
        """
        return torch.round(predicted)

    def save(self,
             model_dir: str | bytes | os.PathLike[str] | os.PathLike[bytes],
             model_name: str) -> None:
        """
        Saves the model state dictionary to a file.

        Args:
            model_dir (str | bytes | os.PathLike[str] | os.PathLike[bytes]): The directory to save the model.
            model_name (str): The name of the model file.
        """
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))

    def load(self,
             model_dir: str | bytes | os.PathLike[str] | os.PathLike[bytes],
             model_name: str) -> Self:
        """
        Loads the model state dictionary from a file.

        Args:
            model_dir (str | bytes | os.PathLike[str] | os.PathLike[bytes]): The directory to load the model from.
            model_name (str): The name of the model file.

        Returns:
            Self: The loaded model.
        """
        self.load_state_dict(torch.load(os.path.join(model_dir, f'{model_name}.pth')))
        return self


