from typing import Sequence, Type

from torch import nn


_activations = [
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
        num_linear_layers (int): The number of linear layers in the module.
        num_inner_layers (int): The number of inner layers in the module.

        linear_num_features (Sequence[int]): A list of the number of features in each layer.
        linear_bias (Sequence[bool]): A list of booleans indicating whether to add a learnable bias to the output.

        activation (Sequence[Type[nn.Module]]): Activation function for each layer.
        activation_kwargs (Sequence[dict[str, any]]): Keyword arguments for the activation function.

        dropout_p (Sequence[float]): Dropout probability for each dropout layer.
        dropout_inplace (Sequence[bool]): Whether to perform dropout in-place.
        dropout_first (Sequence[bool]): Whether to perform dropout before the batch normalization layer.

        batch_norm (Sequence[bool]): Whether to apply batch normalization.
        batch_norm_eps (Sequence[float]): Epsilon value for batch normalization.
        batch_norm_momentum (Sequence[float]): Momentum value for batch normalization.
        batch_norm_affine (Sequence[bool]): Whether to learn affine parameters in batch normalization.
        batch_norm_track_running_stats (Sequence[bool]): Whether to track running statistics in batch normalization.

        modules (nn.Sequential): Sequential container of layers.
    """

    num_linear_layers: int
    num_inner_layers: int

    linear_num_features: Sequence[int]
    linear_bias: Sequence[bool]

    activation: Sequence[Type[nn.Module]]
    activation_kwargs: Sequence[dict[str, any]]

    dropout_p: Sequence[float]
    dropout_inplace: Sequence[bool]
    dropout_first: Sequence[bool]

    batch_norm: Sequence[bool]
    batch_norm_eps: Sequence[float]
    batch_norm_momentum: Sequence[float]
    batch_norm_affine: Sequence[bool]
    batch_norm_track_running_stats: Sequence[bool]

    modules: nn.Sequential

    def __init__(self,
                 num_layers: int,
                 num_features: int | Sequence[int],
                 bias: bool | Sequence[bool] = True,
                 activation: Type[nn.Module] | Sequence[Type[nn.Module]] = nn.ReLU,
                 activation_kwargs: dict[str, any] | Sequence[dict[str, any]] = None,
                 dropout_p: float | Sequence[float] = 0.5,
                 dropout_inplace: bool | Sequence[bool] = True,
                 dropout_first: bool | Sequence[bool] = False,
                 batch_norm: bool | Sequence[bool] = True,
                 batch_norm_eps: float | Sequence[float] = 1e-5,
                 batch_norm_momentum: float | Sequence[float] = 0.1,
                 batch_norm_affine: bool | Sequence[bool] = True,
                 batch_norm_track_running_stats: bool | Sequence[bool] = True) -> None:
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

        self.num_linear_layers = num_layers
        self.num_inner_layers = num_layers - 1

        if isinstance(num_features, int):
            num_features = [num_features] * (self.num_linear_layers + 1)
        if isinstance(bias, bool):
            bias = [bias] * self.num_linear_layers

        if isinstance(activation, type):
            activation = [activation] * self.num_inner_layers
        if isinstance(activation_kwargs, dict):
            activation_kwargs = [activation_kwargs] * self.num_inner_layers
        if activation_kwargs is None:
            activation_kwargs = [{}] * self.num_inner_layers

        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * self.num_inner_layers
        if isinstance(dropout_inplace, bool):
            dropout_inplace = [dropout_inplace] * self.num_inner_layers
        if isinstance(dropout_first, bool):
            dropout_first = [dropout_first] * self.num_inner_layers

        if isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * self.num_inner_layers
        if isinstance(batch_norm_eps, float):
            batch_norm_eps = [batch_norm_eps] * self.num_inner_layers
        if isinstance(batch_norm_momentum, float):
            batch_norm_momentum = [batch_norm_momentum] * self.num_inner_layers
        if isinstance(batch_norm_affine, bool):
            batch_norm_affine = [batch_norm_affine] * self.num_inner_layers
        if isinstance(batch_norm_track_running_stats, bool):
            batch_norm_track_running_stats = [batch_norm_track_running_stats] * self.num_inner_layers

        assert len(num_features) == num_layers + 1, 'linear_num_features must have length num_layers + 1'
        assert len(bias) == num_layers, 'linear_bias must have length num_layers'

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

        self.linear_num_features = num_features
        self.linear_bias = bias

        self.activation = activation
        self.activation_kwargs = activation_kwargs

        self.dropout_p = dropout_p
        self.dropout_inplace = dropout_inplace
        self.dropout_first = dropout_first

        self.batch_norm = batch_norm
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_track_running_stats = batch_norm_track_running_stats

        self.modules = nn.Sequential()

        for i in range(num_layers):
            self.modules.append(nn.Linear(in_features=num_features[i],
                                          out_features=num_features[i + 1],
                                          bias=bias[i]))
            if i >= self.num_inner_layers:
                continue
            self.modules.append(activation[i](**activation_kwargs[i]))
            if dropout_first[i]:
                if dropout_p[i] > 0:
                    self.modules.append(nn.Dropout(p=dropout_p[i],
                                                   inplace=dropout_inplace[i]))
                if batch_norm[i]:
                    self.modules.append(nn.BatchNorm1d(num_features=num_features[i + 1],
                                                       eps=batch_norm_eps[i],
                                                       momentum=batch_norm_momentum[i],
                                                       affine=batch_norm_affine[i],
                                                       track_running_stats=batch_norm_track_running_stats[i]))
            else:
                if batch_norm[i]:
                    self.modules.append(nn.BatchNorm1d(num_features=num_features[i + 1],
                                                       eps=batch_norm_eps[i],
                                                       momentum=batch_norm_momentum[i],
                                                       affine=batch_norm_affine[i],
                                                       track_running_stats=batch_norm_track_running_stats[i]))
                if dropout_p[i] > 0:
                    self.modules.append(nn.Dropout(p=dropout_p[i],
                                                   inplace=dropout_inplace[i]))

    def forward(self, x):
        """
        Forward pass through the module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the layers.
        """
        return self.modules(x)