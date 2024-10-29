import torch.nn as nn
from torch import Tensor

from utils_attacker_lstm import ModelAttackerLSTMLinear


class ModelAttackerConvLSTMLinear(ModelAttackerLSTMLinear):
    """
    A neural network model that combines a 1D convolutional layer with an LSTM and a linear layer.
    Inherits from ModelAttackerLSTMLinear.

    Attributes:
        conv (nn.Conv1d): The 1D convolutional layer.
    """

    conv: nn.Conv1d

    def __init__(self,
                 conv_in_channels: int,
                 conv_out_channels: int,
                 conv_kernel_size: int,
                 conv_stride: int,
                 lstm_hidden_size: int,
                 lstm_num_layers: int = 1,
                 lstm_bidirectional: bool = False,
                 lstm_dropout: float = 0.5,
                 linear_out_features: int = 1):
        """
        Initializes the ModelAttackerConvLSTMLinear.

        Args:
            conv_in_channels (int): Number of input channels for the convolutional layer.
            conv_out_channels (int): Number of output channels for the convolutional layer.
            conv_kernel_size (int): Kernel size for the convolutional layer.
            conv_stride (int): Stride for the convolutional layer.
            lstm_hidden_size (int): Number of features in the hidden state of the LSTM.
            lstm_num_layers (int, optional): Number of recurrent layers in the LSTM. Defaults to 1.
            lstm_bidirectional (bool, optional): If True, the LSTM is bidirectional. Defaults to False.
            lstm_dropout (float, optional): Dropout probability for the LSTM. Defaults to 0.5.
            linear_out_features (int, optional): Number of output features for the linear layer. Defaults to 1.
        """
        super(ModelAttackerConvLSTMLinear, self).__init__(lstm_input_size=conv_out_channels,
                                                          lstm_hidden_size=lstm_hidden_size,
                                                          lstm_num_layers=lstm_num_layers,
                                                          lstm_bidirectional=lstm_bidirectional,
                                                          lstm_dropout=lstm_dropout,
                                                          linear_out_features=linear_out_features)
        self.conv = nn.Conv1d(in_channels=conv_in_channels,
                              out_channels=conv_out_channels,
                              kernel_size=conv_kernel_size,
                              stride=conv_stride)

    def forward(self,
                x: Tensor,
                lstm_hidden: Tensor,
                lstm_cell: Tensor) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, conv_in_channels).
            lstm_hidden (Tensor): Initial hidden state for the LSTM.
            lstm_cell (Tensor): Initial cell state for the LSTM.

        Returns:
            tuple[tuple[Tensor, Tensor], Tensor]: Output from the LSTM and the final hidden and cell states.
        """
        conv_in = x.permute(0, 2, 1) # batch, conv_in_channels, seq_len
        conv_out = self.conv(conv_in)
        lstm_in = conv_out.permute(0, 2, 1) # batch, seq_len, conv_out_channels
        return super().forward(lstm_in, lstm_hidden, lstm_cell)

    @property
    def conv_in_channels(self) -> int:
        """
        Returns the number of input channels for the convolutional layer.

        Returns:
            int: Number of input channels.
        """
        return self.conv.in_channels

    @property
    def conv_out_channels(self) -> int:
        """
        Returns the number of output channels for the convolutional layer.

        Returns:
            int: Number of output channels.
        """
        return self.conv.out_channels

    @property
    def conv_kernel_size(self) -> int:
        """
        Returns the kernel size for the convolutional layer.

        Returns:
            int: Kernel size.
        """
        return self.conv.kernel_size[0]

    @property
    def conv_stride(self) -> int:
        """
        Returns the stride for the convolutional layer.

        Returns:
            int: Stride.
        """
        return self.conv.stride[0]
