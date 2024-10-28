import torch.nn as nn
from torch import Tensor

from utils_attacker_lstm import ModelAttackerLSTMLinear


class ModelAttackerConvLSTMLinear(ModelAttackerLSTMLinear):

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
        conv_in = x.permute(0, 2, 1) # batch, conv_in_channels, seq_len
        conv_out = self.conv(conv_in)
        lstm_in = conv_out.permute(0, 2, 1) # batch, seq_len, conv_out_channels
        return super().forward(lstm_in, lstm_hidden, lstm_cell)

    @property
    def conv_in_channels(self) -> int:
        return self.conv.in_channels

    @property
    def conv_out_channels(self) -> int:
        return self.conv.out_channels

    @property
    def conv_kernel_size(self) -> int:
        return self.conv.kernel_size[0]

    @property
    def conv_stride(self) -> int:
        return self.conv.stride[0]
