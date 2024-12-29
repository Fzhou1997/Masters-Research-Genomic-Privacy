import os
from os import PathLike
from typing import Type, Sequence, Self

import torch
from torch import nn, Tensor

from src.utils_torch.modules import Conv1dStack, LSTMStack, LSTMStackHiddenCell, LinearStack

_activations = {
    "Identity",
    "Threshold",
    "ReLU",
    "RReLU",
    "Hardtanh",
    "ReLU6",
    "Sigmoid",
    "Hardsigmoid",
    "Tanh",
    "SiLU",
    "Mish",
    "Hardswish",
    "ELU",
    "CELU",
    "SELU",
    "GLU",
    "GELU",
    "Hardshrink",
    "LeakyReLU",
    "LogSigmoid",
    "Softplus",
    "Softshrink",
    "MultiheadAttention",
    "PReLU",
    "Softsign",
    "Tanhshrink",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax"
}


class ModelAttackerLSTM(nn.Module):
    _conv_lstm_activation: Type[nn.Module]
    _conv_lstm_activation_kwargs: dict[str, any]

    _conv_lstm_dropout_p: float
    _conv_lstm_dropout_first: bool

    _conv_lstm_layer_norm: bool

    _lstm_linear_dropout_p: float
    _lstm_linear_dropout_first: bool

    _lstm_linear_batch_norm: bool
    _lstm_linear_batch_norm_momentum: float

    _hidden_cell_mode: bool

    _conv_stack: Conv1dStack | nn.Identity

    _conv_lstm_activation_module: nn.Module
    _conv_lstm_dropout_module: nn.Dropout | nn.Identity
    _conv_lstm_layer_norm_module: nn.LayerNorm | nn.Identity

    _lstm_stack: LSTMStack | LSTMStackHiddenCell

    _lstm_linear_dropout_module: nn.Dropout | nn.Identity
    _lstm_linear_batch_norm_module: nn.BatchNorm1d | nn.Identity

    _linear_stack: LinearStack

    def __init__(self,
                 conv_num_layers: int,
                 conv_channel_size: int | Sequence[int],
                 conv_kernel_size: int | Sequence[int],
                 lstm_num_layers: int,
                 lstm_input_size: int,
                 lstm_hidden_size: int | Sequence[int],
                 linear_num_layers: int,
                 linear_num_features: int | Sequence[int],
                 conv_stride: int | Sequence[int] = 1,
                 conv_dilation: int | Sequence[int] = 1,
                 conv_groups: int | Sequence[int] = 1,
                 conv_activation: Type[nn.Module] | Sequence[Type[nn.Module]] = nn.ReLU,
                 conv_activation_kwargs: dict[str, any] | Sequence[dict[str, any]] = None,
                 conv_dropout_p: float | Sequence[float] = 0.5,
                 conv_dropout_first: bool | Sequence[bool] = True,
                 conv_batch_norm: bool | Sequence[bool] = True,
                 conv_batch_norm_momentum: float | Sequence[float] = 0.1,
                 conv_lstm_activation: Type[nn.Module] = nn.ReLU,
                 conv_lstm_activation_kwargs: dict[str, any] = None,
                 conv_lstm_dropout_p: float = 0.5,
                 conv_lstm_dropout_first: bool = True,
                 conv_lstm_layer_norm: bool = True,
                 lstm_proj_size: int | Sequence[int] = 0,
                 lstm_bidirectional: bool | Sequence[bool] = False,
                 lstm_dropout_p: float | Sequence[float] = 0.5,
                 lstm_dropout_first: bool | Sequence[bool] = True,
                 lstm_layer_norm: bool | Sequence[bool] = True,
                 lstm_linear_dropout_p: float = 0.25,
                 lstm_linear_dropout_first: bool = True,
                 lstm_linear_batch_norm: bool = True,
                 lstm_linear_batch_norm_momentum: float = 0.1,
                 linear_activation: Type[nn.Module] = nn.ReLU,
                 linear_activation_kwargs: dict[str, any] = None,
                 linear_dropout_p: float = 0.5,
                 linear_dropout_first: bool = True,
                 linear_batch_norm: bool = True,
                 linear_batch_norm_momentum: float = 0.1,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        super(ModelAttackerLSTM, self).__init__()

        assert conv_num_layers >= 0, "conv_num_layers must be non-negative"
        assert lstm_num_layers > 0, "lstm_num_layers must be positive"
        assert linear_num_layers > 0, "linear_num_layers must be positive"

        self._conv_lstm_activation = conv_lstm_activation
        self._conv_lstm_activation_kwargs = conv_lstm_activation_kwargs if conv_lstm_activation_kwargs is not None else {}

        self._conv_lstm_dropout_p = conv_lstm_dropout_p
        self._conv_lstm_dropout_first = conv_lstm_dropout_first

        self._conv_lstm_layer_norm = conv_lstm_layer_norm

        if conv_num_layers == 0:
            self._conv_stack = nn.Identity()
            self._conv_lstm_activation_module = nn.Identity()
            self._conv_lstm_dropout_module = nn.Identity()
            self._conv_lstm_layer_norm_module = nn.Identity()
        else:
            self._conv_stack = Conv1dStack(num_layers=conv_num_layers,
                                           channel_size=conv_channel_size,
                                           kernel_size=conv_kernel_size,
                                           stride=conv_stride,
                                           dilation=conv_dilation,
                                           groups=conv_groups,
                                           activation=conv_activation,
                                           activation_kwargs=conv_activation_kwargs,
                                           dropout_p=conv_dropout_p,
                                           dropout_first=conv_dropout_first,
                                           batch_norm=conv_batch_norm,
                                           batch_norm_momentum=conv_batch_norm_momentum,
                                           device=device,
                                           dtype=dtype)
            self._conv_lstm_activation_module = conv_lstm_activation(**conv_lstm_activation_kwargs)
            if conv_lstm_dropout_p > 0:
                self._conv_lstm_dropout_module = nn.Dropout(p=conv_lstm_dropout_p)
            else:
                self._conv_lstm_dropout_module = nn.Identity()
            if conv_lstm_layer_norm:
                self._conv_lstm_layer_norm_module = nn.LayerNorm(
                    normalized_shape=self._conv_stack.conv_channel_size_out,
                    device=device,
                    dtype=dtype)
            else:
                self._conv_lstm_layer_norm_module = nn.Identity()

        self._lstm_stack = LSTMStack(num_layers=lstm_num_layers,
                                     input_size=lstm_input_size,
                                     hidden_size=lstm_hidden_size,
                                     proj_size=lstm_proj_size,
                                     bidirectional=lstm_bidirectional,
                                     dropout_p=lstm_dropout_p,
                                     dropout_first=lstm_dropout_first,
                                     layer_norm=lstm_layer_norm,
                                     device=device,
                                     dtype=dtype)

        self._lstm_linear_dropout_p = lstm_linear_dropout_p
        self._lstm_linear_dropout_first = lstm_linear_dropout_first

        self._lstm_linear_batch_norm = lstm_linear_batch_norm
        self._lstm_linear_batch_norm_momentum = lstm_linear_batch_norm_momentum

        if lstm_linear_dropout_p > 0:
            self._lstm_linear_dropout_module = nn.Dropout(p=lstm_linear_dropout_p)
        else:
            self._lstm_linear_dropout_module = nn.Identity()
        if lstm_linear_batch_norm:
            self._lstm_linear_batch_norm_module = nn.BatchNorm1d(
                num_features=self._lstm_stack.lstm_output_size_out,
                momentum=lstm_linear_batch_norm_momentum,
                device=device,
                dtype=dtype)
        else:
            self._lstm_linear_batch_norm_module = nn.Identity()

        self._linear_stack = LinearStack(num_layers=linear_num_layers,
                                         num_features=linear_num_features,
                                         activation=linear_activation,
                                         activation_kwargs=linear_activation_kwargs,
                                         dropout_p=linear_dropout_p,
                                         dropout_first=linear_dropout_first,
                                         batch_norm=linear_batch_norm,
                                         batch_norm_momentum=linear_batch_norm_momentum,
                                         device=device,
                                         dtype=dtype)

        if conv_num_layers > 0:
            assert self._conv_stack.conv_channel_size_out == self._lstm_stack.lstm_input_size_in, "conv_channel_size_out must be equal to lstm_input_size_in"
        assert self._lstm_stack.lstm_output_size_out == self._linear_stack.linear_num_features_in, "lstm_output_size_out must be equal to linear_num_features_in"
        assert self._linear_stack.linear_num_features_out == 1, "linear_num_features_out must be equal to 1"

        self._hidden_cell_mode = False

    @property
    def conv_num_layers(self) -> int:
        if isinstance(self._conv_stack, nn.Identity):
            return 0
        return self._conv_stack.conv_num_layers

    @property
    def conv_channel_size_in(self) -> int:
        if isinstance(self._conv_stack, nn.Identity):
            return 0
        return self._conv_stack.conv_channel_size_in

    @property
    def conv_channel_size_out(self) -> int:
        if isinstance(self._conv_stack, nn.Identity):
            return 0
        return self._conv_stack.conv_channel_size_out

    @property
    def conv_channel_size(self) -> tuple[int, ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_channel_size

    @property
    def conv_kernel_size(self) -> tuple[int, ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_kernel_size

    @property
    def conv_stride(self) -> tuple[int, ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_stride

    @property
    def conv_dilation(self) -> tuple[int, ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_dilation

    @property
    def conv_groups(self) -> tuple[int, ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_groups

    @property
    def conv_activation(self) -> tuple[Type[nn.Module], ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_activation

    @property
    def conv_activation_kwargs(self) -> tuple[dict[str, any], ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_activation_kwargs

    @property
    def conv_dropout_p(self) -> tuple[float, ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_dropout_p

    @property
    def conv_dropout_first(self) -> tuple[bool, ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_dropout_first

    @property
    def conv_batch_norm(self) -> tuple[bool, ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_batch_norm

    @property
    def conv_batch_norm_momentum(self) -> tuple[float, ...]:
        if isinstance(self._conv_stack, nn.Identity):
            return ()
        return self._conv_stack.conv_batch_norm_momentum

    @property
    def conv_lstm_activation(self) -> Type[nn.Module]:
        return self._conv_lstm_activation

    @property
    def conv_lstm_activation_kwargs(self) -> dict[str, any]:
        return self._conv_lstm_activation_kwargs

    @property
    def conv_lstm_dropout_p(self) -> float:
        return self._conv_lstm_dropout_p

    @property
    def conv_lstm_dropout_first(self) -> bool:
        return self._conv_lstm_dropout_first

    @property
    def conv_lstm_layer_norm(self) -> bool:
        return self._conv_lstm_layer_norm

    @property
    def lstm_num_layers(self) -> int:
        return self._lstm_stack.lstm_num_layers

    @property
    def lstm_input_size_in(self) -> int:
        return self._lstm_stack.lstm_input_size_in

    @property
    def lstm_output_size_out(self) -> int:
        return self._lstm_stack.lstm_output_size_out

    @property
    def lstm_input_size(self) -> tuple[int, ...]:
        return self._lstm_stack.lstm_input_size

    @property
    def lstm_hidden_size(self) -> tuple[int, ...]:
        return self._lstm_stack.lstm_hidden_size

    @property
    def lstm_proj_size(self) -> tuple[int, ...]:
        return self._lstm_stack.lstm_proj_size

    @property
    def lstm_output_size(self) -> tuple[int, ...]:
        return self._lstm_stack.lstm_output_size

    @property
    def lstm_bidirectional(self) -> tuple[bool, ...]:
        return self._lstm_stack.lstm_bidirectional

    @property
    def lstm_num_directions(self) -> tuple[int, ...]:
        return self._lstm_stack.lstm_num_directions

    @property
    def lstm_dropout_p(self) -> tuple[float, ...]:
        return self._lstm_stack.lstm_dropout_p

    @property
    def lstm_dropout_first(self) -> tuple[bool, ...]:
        return self._lstm_stack.lstm_dropout_first

    @property
    def lstm_layer_norm(self) -> tuple[bool, ...]:
        return self._lstm_stack.lstm_layer_norm

    @property
    def lstm_linear_dropout_p(self) -> float:
        return self._lstm_linear_dropout_p

    @property
    def lstm_linear_dropout_first(self) -> bool:
        return self._lstm_linear_dropout_first

    @property
    def lstm_linear_batch_norm(self) -> bool:
        return self._lstm_linear_batch_norm

    @property
    def lstm_linear_batch_norm_momentum(self) -> float:
        return self._lstm_linear_batch_norm_momentum

    @property
    def linear_num_layers(self) -> int:
        return self._linear_stack.linear_num_layers

    @property
    def linear_num_features_in(self) -> int:
        return self._linear_stack.linear_num_features_in

    @property
    def linear_num_features_out(self) -> int:
        return self._linear_stack.linear_num_features_out

    @property
    def linear_num_features(self) -> tuple[int, ...]:
        return self._linear_stack.linear_num_features

    @property
    def linear_activation(self) -> tuple[Type[nn.Module], ...]:
        return self._linear_stack.linear_activation

    @property
    def linear_activation_kwargs(self) -> tuple[dict[str, any], ...]:
        return self._linear_stack.linear_activation_kwargs

    @property
    def linear_dropout_p(self) -> tuple[float, ...]:
        return self._linear_stack.linear_dropout_p

    @property
    def linear_dropout_first(self) -> tuple[bool, ...]:
        return self._linear_stack.linear_dropout_first

    @property
    def linear_batch_norm(self) -> tuple[bool, ...]:
        return self._linear_stack.linear_batch_norm

    @property
    def linear_batch_norm_momentum(self) -> tuple[float, ...]:
        return self._linear_stack.linear_batch_norm_momentum

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_hx(self, batch_size: int) -> tuple[tuple[Tensor, Tensor], ...]:
        return self._lstm_stack.get_hx(batch_size)

    def forward(self, x: Tensor, hx: tuple[tuple[Tensor, Tensor], ...] = None) -> tuple[Tensor, tuple[tuple[Tensor | tuple[Tensor, Tensor], tuple[Tensor, Tensor]], ...]]:
        if self.conv_num_layers > 0:
            x = x.permute(0, 2, 1)
            x = self._conv_stack(x)
            x = self._conv_lstm_activation_module(x)
            if self._conv_lstm_dropout_first:
                x = self._conv_lstm_dropout_module(x)
                x = x.permute(0, 2, 1)
                x = self._conv_lstm_layer_norm_module(x)
            else:
                x = x.permute(0, 2, 1)
                x = self._conv_lstm_layer_norm_module(x)
                x = self._conv_lstm_dropout_module(x)
        out = self._lstm_stack.forward(x, hx)
        out_last = out[-1]
        _, (h_out, c_out) = out_last
        if self.lstm_bidirectional[-1]:
            x = torch.cat((h_out[-2], h_out[-1]), dim=-1)
        else:
            x = h_out[-1]
        if self._lstm_linear_dropout_first:
            x = self._lstm_linear_dropout_module(x)
            x = self._lstm_linear_batch_norm_module(x)
        else:
            x = self._lstm_linear_batch_norm_module(x)
            x = self._lstm_linear_dropout_module(x)
        x = self._linear_stack(x).squeeze(-1)
        return x, out

    def predict(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits)

    def classify(self, predicted: Tensor) -> Tensor:
        return torch.round(predicted)

    def set_hidden_cell_mode(self, hidden_cell_model: bool) -> None:
        if hidden_cell_model == self._hidden_cell_mode:
            return
        lstm_stack = self._lstm_stack
        if not hidden_cell_model:
            self._lstm_stack = LSTMStack(num_layers=lstm_stack.lstm_num_layers,
                                         input_size=lstm_stack.lstm_input_size_in,
                                         hidden_size=lstm_stack.lstm_hidden_size,
                                         proj_size=lstm_stack.lstm_proj_size,
                                         bidirectional=lstm_stack.lstm_bidirectional,
                                         dropout_p=lstm_stack.lstm_dropout_p,
                                         dropout_first=lstm_stack.lstm_dropout_first,
                                         layer_norm=lstm_stack.lstm_layer_norm,
                                         device=lstm_stack.device,
                                         dtype=lstm_stack.dtype)
        else:
            self._lstm_stack = LSTMStackHiddenCell(num_layers=lstm_stack.lstm_num_layers,
                                                   input_size=lstm_stack.lstm_input_size_in,
                                                   hidden_size=lstm_stack.lstm_hidden_size,
                                                   proj_size=lstm_stack.lstm_proj_size,
                                                   bidirectional=lstm_stack.lstm_bidirectional,
                                                   dropout_p=lstm_stack.lstm_dropout_p,
                                                   dropout_first=lstm_stack.lstm_dropout_first,
                                                   layer_norm=lstm_stack.lstm_layer_norm,
                                                   device=lstm_stack.device,
                                                   dtype=lstm_stack.dtype)
        self._lstm_stack.set_parameters(lstm_stack.get_parameters())
        self._hidden_cell_mode = hidden_cell_model

    def save(self,
             model_dir: str | bytes | PathLike[str] | PathLike[bytes],
             model_name: str) -> None:
        prev_mode = self._hidden_cell_mode
        self.set_hidden_cell_mode(False)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))
        self.set_hidden_cell_mode(prev_mode)

    def load(self,
             model_dir: str | bytes | PathLike[str] | PathLike[bytes],
             model_name: str) -> Self:
        prev_mode = self._hidden_cell_mode
        self.set_hidden_cell_mode(False)
        self.load_state_dict(torch.load(os.path.join(model_dir, f'{model_name}.pth')))
        self.set_hidden_cell_mode(prev_mode)
        return self
