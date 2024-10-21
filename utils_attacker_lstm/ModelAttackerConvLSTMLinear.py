import torch.nn as nn

from utils_attacker_lstm import ModelAttackerLSTMLinear


class ModelAttackerConvLSTMLinear(ModelAttackerLSTMLinear):

    conv: nn.Conv1d

    def __init__(self,
                 conv_input_channels: int,
                 conv_output_channels: int,
                 conv_kernel_size: int,
                 conv_stride: int,
                 lstm_hidden_size: int,
                 lstm_num_layers: int = 1,
                 lstm_bidirectional: bool = False,
                 lstm_dropout: float = 0.5,
                 linear_output_size: int = 1):
        self.conv = nn.Conv1d(in_channels=conv_input_channels,
                              out_channels=conv_output_channels,
                              kernel_size=conv_kernel_size,
                              stride=conv_stride)
        super(ModelAttackerConvLSTMLinear, self).__init__(lstm_input_size=conv_output_channels,
                                                          lstm_hidden_size=lstm_hidden_size,
                                                          lstm_num_layers=lstm_num_layers,
                                                          lstm_bidirectional=lstm_bidirectional,
                                                          lstm_dropout=lstm_dropout,
                                                          linear_output_size=linear_output_size)

    @property
    def conv_input_channels(self) -> int:
        return self.conv.in_channels

    @property
    def conv_output_channels(self) -> int:
        return self.conv.out_channels

    @property
    def conv_kernel_size(self) -> int:
        return self.conv.kernel_size[0]

    @property
    def conv_stride(self) -> int:
        return self.conv.stride[0]
