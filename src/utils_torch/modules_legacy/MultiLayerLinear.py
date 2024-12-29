from typing import Sequence, Type

import torch
from torch import nn, Tensor

_activations = [
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
    "LogSoftmax",
]

class MultiLayerLinear(nn.Module):
    """
    A multi-layer linear module with optional batch normalization and dropout.

    Attributes:
        _linear_num_layers (int): The number of linear layers in the module.
        _inner_num_layers (int): The number of inner layers in the module.

        _linear_num_features (tuple[int]): A list of the number of features in each layer.
        _linear_bias (tuple[bool]): A list of booleans indicating whether to add a learnable bias to the output.

        _activation (tuple[Type[nn.Module]]): Activation function for each layer.
        _activation_kwargs (tuple[dict[str, any]]): Keyword arguments for the activation function.

        _dropout_p (tuple[float]): Dropout probability for each dropout layer.
        _dropout_inplace (tuple[bool]): Whether to perform dropout in-place.
        _dropout_first (tuple[bool]): Whether to perform dropout before the batch normalization layer.

        _batch_norm (tuple[bool]): Whether to apply batch normalization.
        _batch_norm_eps (tuple[float]): Epsilon value for batch normalization.
        _batch_norm_momentum (tuple[float]): Momentum value for batch normalization.
        _batch_norm_affine (tuple[bool]): Whether to learn affine parameters in batch normalization.
        _batch_norm_track_running_stats (tuple[bool]): Whether to track running statistics in batch normalization.

        _multi_layer_modules (nn.Sequential): Sequential container of layers.
    """

    _linear_num_layers: int
    _inner_num_layers: int

    _linear_num_features: tuple[int, ...]
    _linear_bias: tuple[bool, ...]

    _activation: tuple[Type[nn.Module], ...]
    _activation_kwargs: tuple[dict[str, any], ...]

    _dropout_p: tuple[float, ...]
    _dropout_inplace: tuple[bool, ...]
    _dropout_first: tuple[bool, ...]

    _batch_norm: tuple[bool, ...]
    _batch_norm_eps: tuple[float, ...]
    _batch_norm_momentum: tuple[float, ...]
    _batch_norm_affine: tuple[bool, ...]
    _batch_norm_track_running_stats: tuple[bool, ...]

    _multi_layer_modules: nn.Sequential

    def __init__(self,
                 num_layers: int,
                 num_features: int | Sequence[int],
                 bias: bool | Sequence[bool] = True,
                 activation: Type[nn.Module] | Sequence[Type[nn.Module]] = nn.ReLU,
                 activation_kwargs: dict[str, any] | Sequence[dict[str, any]] = None,
                 dropout_p: float | Sequence[float] = 0.5,
                 dropout_inplace: bool | Sequence[bool] = False,
                 dropout_first: bool | Sequence[bool] = True,
                 batch_norm: bool | Sequence[bool] = True,
                 batch_norm_eps: float | Sequence[float] = 1e-5,
                 batch_norm_momentum: float | Sequence[float] = 0.1,
                 batch_norm_affine: bool | Sequence[bool] = True,
                 batch_norm_track_running_stats: bool | Sequence[bool] = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes the multi-layer linear module.

        Args:
            num_layers (int): The number of linear layers in the module.
            num_features (int | Sequence[int]): The number of features in each layer.
            bias (bool | Sequence[bool], optional): Whether to use bias in each layer, default is True.
            activation (Type[nn.Module] | Sequence[Type[nn.Module]], optional): Activation function for each layer, default is nn.ReLU.
            activation_kwargs (dict[str, any] | Sequence[dict[str, any]], optional): Keyword arguments for the activation function, default is None.
            dropout_p (float | Sequence[float], optional): Dropout probability for each dropout layer, default is 0.5.
            dropout_inplace (bool | Sequence[bool], optional): Whether to perform dropout in-place, default is False.
            dropout_first (bool | Sequence[bool], optional): Whether to perform dropout before the batch normalization layer, default is True.
            batch_norm (bool | Sequence[bool], optional): Whether to apply batch normalization, default is True.
            batch_norm_eps (float | Sequence[float], optional): Epsilon value for batch normalization, default is 1e-5.
            batch_norm_momentum (float | Sequence[float], optional): Momentum value for batch normalization, default is 0.1.
            batch_norm_affine (bool | Sequence[bool], optional): Whether to learn affine parameters in batch normalization, default is True.
            batch_norm_track_running_stats (bool | Sequence[bool], optional): Whether to track running statistics in batch normalization, default is True.
            device (torch.device, optional): The device for the module, default is None.
            dtype (torch.dtype, optional): The data type for the module, default is None.

        Raises:
            AssertionError: If num_layers is less than or equal to 0.
            AssertionError: If num_features does not have length num_layers + 1.
            AssertionError: If bias does not have length num_layers.
            AssertionError: If activation, activation_kwargs do not have length num_layers - 1.
            AssertionError: If activation is not one of the supported activation functions.
            AssertionError: If dropout_p, dropout_inplace do not have length num_layers - 1.
            AssertionError: If batch_norm, batch_norm_eps, batch_norm_momentum, batch_norm_affine, and batch_norm_track_running_stats do not have length num_layers - 1.
        """
        super(MultiLayerLinear, self).__init__()

        assert num_layers > 0, 'num_layers must be greater than 0'

        self._linear_num_layers = num_layers
        self._inner_num_layers = num_layers - 1

        if isinstance(num_features, int):
            num_features = [num_features] * (self._linear_num_layers + 1)
        if isinstance(bias, bool):
            bias = [bias] * self._linear_num_layers

        if isinstance(activation, type):
            activation = [activation] * self._inner_num_layers
        if isinstance(activation_kwargs, dict):
            activation_kwargs = [activation_kwargs] * self._inner_num_layers
        if activation is None:
            activation = [nn.Identity] * self._inner_num_layers
        if activation_kwargs is None:
            activation_kwargs = [{}] * self._inner_num_layers

        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * self._inner_num_layers
        if isinstance(dropout_inplace, bool):
            dropout_inplace = [dropout_inplace] * self._inner_num_layers
        if isinstance(dropout_first, bool):
            dropout_first = [dropout_first] * self._inner_num_layers

        if isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * self._inner_num_layers
        if isinstance(batch_norm_eps, float):
            batch_norm_eps = [batch_norm_eps] * self._inner_num_layers
        if isinstance(batch_norm_momentum, float):
            batch_norm_momentum = [batch_norm_momentum] * self._inner_num_layers
        if isinstance(batch_norm_affine, bool):
            batch_norm_affine = [batch_norm_affine] * self._inner_num_layers
        if isinstance(batch_norm_track_running_stats, bool):
            batch_norm_track_running_stats = [batch_norm_track_running_stats] * self._inner_num_layers

        assert len(num_features) == num_layers + 1, 'num_features must have length num_layers + 1'
        assert len(bias) == num_layers, 'bias must have length num_layers'

        assert len(activation) == num_layers - 1, 'activation must have length num_layers - 1'
        assert len(activation_kwargs) == num_layers - 1, 'activation_kwargs must have length num_layers - 1'
        assert all([cls is None or cls.__name__ in _activations for cls in activation]), 'activation must be one of the following types: ' + ', '.join(_activations)

        assert len(dropout_p) == num_layers - 1, 'dropout_p must have length num_layers - 1'
        assert len(dropout_inplace) == num_layers - 1, 'dropout_inplace must have length num_layers - 1'
        assert len(dropout_first) == num_layers - 1, 'dropout_first must have length num_layers - 1'

        assert len(batch_norm) == num_layers - 1, 'batch_norm must have length num_layers - 1'
        assert len(batch_norm_eps) == num_layers - 1, 'batch_norm_eps must have length num_layers - 1'
        assert len(batch_norm_momentum) == num_layers - 1, 'batch_norm_momentum must have length num_layers - 1'
        assert len(batch_norm_affine) == num_layers - 1, 'batch_norm_affine must have length num_layers - 1'
        assert len(batch_norm_track_running_stats) == num_layers - 1, 'batch_norm_track_running_stats must have length num_layers - 1'

        self._linear_num_features = tuple(num_features)
        self._linear_bias = tuple(bias)

        for i in range(self._inner_num_layers):
            if activation[i] is None:
                activation[i] = nn.Identity
            if activation_kwargs[i] is None:
                activation_kwargs[i] = {}
        self._activation = tuple(activation)
        self._activation_kwargs = tuple(activation_kwargs)

        self._dropout_p = tuple(dropout_p)
        self._dropout_inplace = tuple(dropout_inplace)
        self._dropout_first = tuple(dropout_first)

        self._batch_norm = tuple(batch_norm)
        self._batch_norm_eps = tuple(batch_norm_eps)
        self._batch_norm_momentum = tuple(batch_norm_momentum)
        self._batch_norm_affine = tuple(batch_norm_affine)
        self._batch_norm_track_running_stats = tuple(batch_norm_track_running_stats)

        self._multi_layer_modules = nn.Sequential()

        for i in range(num_layers):
            self._multi_layer_modules.append(nn.Linear(in_features=num_features[i],
                                                       out_features=num_features[i + 1],
                                                       bias=bias[i],
                                                       device=device,
                                                       dtype=dtype))
            if i >= self._inner_num_layers:
                continue
            self._multi_layer_modules.append(self._activation[i](**self._activation_kwargs[i]))
            if dropout_first[i]:
                if dropout_p[i] > 0:
                    self._multi_layer_modules.append(nn.Dropout(p=dropout_p[i],
                                                                inplace=dropout_inplace[i]))
                if batch_norm[i]:
                    self._multi_layer_modules.append(nn.BatchNorm1d(num_features=num_features[i + 1],
                                                                    eps=batch_norm_eps[i],
                                                                    momentum=batch_norm_momentum[i],
                                                                    affine=batch_norm_affine[i],
                                                                    track_running_stats=batch_norm_track_running_stats[i],
                                                                    device=device,
                                                                    dtype=dtype))
            else:
                if batch_norm[i]:
                    self._multi_layer_modules.append(nn.BatchNorm1d(num_features=num_features[i + 1],
                                                                    eps=batch_norm_eps[i],
                                                                    momentum=batch_norm_momentum[i],
                                                                    affine=batch_norm_affine[i],
                                                                    track_running_stats=batch_norm_track_running_stats[i],
                                                                    device=device,
                                                                    dtype=dtype))
                if dropout_p[i] > 0:
                    self._multi_layer_modules.append(nn.Dropout(p=dropout_p[i],
                                                                inplace=dropout_inplace[i]))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the layers.
        """
        return self._multi_layer_modules(x)

    @property
    def linear_num_layers(self) -> int:
        """
        Returns the number of linear layers.

        Returns:
            int: Number of linear layers.
        """
        return self._linear_num_layers

    @property
    def linear_num_features_in(self) -> int:
        """
        Returns the number of input features to the first linear layer.

        Returns:
            int: Number of input features to the first linear layer.
        """
        return self._linear_num_features[0]

    @property
    def linear_num_features_out(self) -> int:
        """
        Returns the number of output features from the last linear layer.

        Returns:
            int: Number of output features from the last linear layer.
        """
        return self._linear_num_features[-1]

    @property
    def linear_num_features(self) -> tuple[int, ...]:
        """
        Returns the number of features in each layer.

        Returns:
            tuple[int, ...]: Number of features in each layer.
        """
        return self._linear_num_features

    @property
    def linear_bias(self) -> tuple[bool, ...]:
        """
        Returns whether bias is used in each layer.

        Returns:
            tuple[bool, ...]: Whether bias is used in each layer.
        """
        return self._linear_bias

    @property
    def linear_activation(self) -> tuple[Type[nn.Module], ...]:
        """
        Returns the activation function for each layer.

        Returns:
            tuple[Type[nn.Module], ...]: Activation function for each layer.
        """
        return self._activation

    @property
    def linear_activation_kwargs(self) -> tuple[dict[str, any], ...]:
        """
        Returns the keyword arguments for the activation function.

        Returns:
            tuple[dict[str, any], ...]: Keyword arguments for the activation function.
        """
        return self._activation_kwargs

    @property
    def linear_dropout_p(self) -> tuple[float, ...]:
        """
        Returns the dropout probability for each dropout layer.

        Returns:
            tuple[float, ...]: Dropout probability for each dropout layer.
        """
        return self._dropout_p

    @property
    def linear_dropout_inplace(self) -> tuple[bool, ...]:
        """
        Returns whether dropout is performed in-place.

        Returns:
            tuple[bool, ...]: Whether dropout is performed in-place.
        """
        return self._dropout_inplace

    @property
    def linear_dropout_first(self) -> tuple[bool, ...]:
        """
        Returns whether dropout is performed before batch normalization.

        Returns:
            tuple[bool, ...]: Whether dropout is performed before batch normalization.
        """
        return self._dropout_first

    @property
    def linear_batch_norm(self) -> tuple[bool, ...]:
        """
        Returns whether batch normalization is applied.

        Returns:
            tuple[bool, ...]: Whether batch normalization is applied.
        """
        return self._batch_norm

    @property
    def linear_batch_norm_eps(self) -> tuple[float, ...]:
        """
        Returns the epsilon value for batch normalization.

        Returns:
            tuple[float, ...]: Epsilon value for batch normalization.
        """
        return self._batch_norm_eps

    @property
    def linear_batch_norm_momentum(self) -> tuple[float, ...]:
        """
        Returns the momentum value for batch normalization.

        Returns:
            tuple[float, ...]: Momentum value for batch normalization.
        """
        return self._batch_norm_momentum

    @property
    def linear_batch_norm_affine(self) -> tuple[bool, ...]:
        """
        Returns whether affine parameters are learned in batch normalization.

        Returns:
            tuple[bool, ...]: Whether affine parameters are learned in batch normalization.
        """
        return self._batch_norm_affine

    @property
    def linear_batch_norm_track_running_stats(self) -> tuple[bool, ...]:
        """
        Returns whether running statistics are tracked in batch normalization.

        Returns:
            tuple[bool, ...]: Whether running statistics are tracked in batch normalization.
        """
        return self._batch_norm_track_running_stats