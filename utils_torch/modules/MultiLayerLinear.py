from typing import Sequence

from torch import nn


class MultiLayerLinear(nn.Module):
    """
    A multi-layer linear module with optional batch normalization and dropout.

    Attributes:
        num_batch_norm_layers (int): The number of batch normalization layers in the module.
        batch_norm (Sequence[bool]): A list of booleans indicating whether to use batch normalization for each layer.
        batch_norm_eps (Sequence[float]): A list of epsilon values added to the denominator for numerical stability.
        batch_norm_momentum (Sequence[float]): A list of values used for the running_mean and running_var computation.
        batch_norm_affine (Sequence[bool]): A list of booleans indicating whether the module has learnable affine parameters.
        batch_norm_track_running_stats (Sequence[bool]): A list of booleans indicating whether the module tracks the running mean and variance.

        num_linear_layers (int): The number of linear layers in the module.
        linear_num_features (Sequence[int]): A list of the number of features in each layer.
        linear_bias (Sequence[bool]): A list of booleans indicating whether to add a learnable bias to the output.

        num_dropout_layers (int): The number of dropout layers in the module.
        dropout_p (Sequence[float]): A list of dropout probabilities for each layer.
        dropout_inplace (Sequence[bool]): A list of booleans indicating whether to operate in-place.

        modules (nn.Sequential): The sequential container for the module.
    """

    num_batch_norm_layers: int
    batch_norm: Sequence[bool]
    batch_norm_eps: Sequence[float]
    batch_norm_momentum: Sequence[float]
    batch_norm_affine: Sequence[bool]
    batch_norm_track_running_stats: Sequence[bool]

    num_linear_layers: int
    linear_num_features: Sequence[int]
    linear_bias: Sequence[bool]

    num_dropout_layers: int
    dropout_p: Sequence[float]
    dropout_inplace: Sequence[bool]

    modules = nn.Sequential

    def __init__(self,
                 num_layers: int,
                 num_features: int | Sequence[int],
                 bias: bool | Sequence[bool],
                 dropout_p: float | Sequence[float],
                 dropout_inplace: bool | Sequence[bool],
                 batch_norm: bool | Sequence[bool],
                 batch_norm_eps: float | Sequence[float],
                 batch_norm_momentum: float | Sequence[float],
                 batch_norm_affine: bool | Sequence[bool],
                 batch_norm_track_running_stats: bool | Sequence[bool]):
        """
        Initializes the multi-layer linear module.

        Args:
            num_layers (int): The number of linear layers in the module.
            num_features (int | Sequence[int]): The number of features in each layer.
            bias (bool | Sequence[bool]): If True, adds a learnable bias to the output.
            dropout_p (float | Sequence[float]): If non-zero, introduces a Dropout layer on the outputs of each linear layer except the last layer.
            dropout_inplace (bool | Sequence[bool]): If True, operates in-place.
            batch_norm (bool | Sequence[bool]): If True, adds a BatchNorm1d layer before each linear layer.
            batch_norm_eps (float | Sequence[float]): A value added to the denominator for numerical stability.
            batch_norm_momentum (float | Sequence[float]): The value used for the running_mean and running_var computation.
            batch_norm_affine (bool | Sequence[bool]): If True, this module has learnable affine parameters.
            batch_norm_track_running_stats (bool | Sequence[bool]): If True, this module tracks the running mean and variance.

        Raises:
            AssertionError: If num_layers is less than or equal to 0.
            AssertionError: If batch_norm, batch_norm_eps, batch_norm_momentum, batch_norm_affine, and batch_norm_track_running_stats do not have length num_layers.
            AssertionError: If num_features does not have length num_layers + 1.
            AssertionError: If bias does not have length num_layers.
            AssertionError: If dropout_p, dropout_inplace does not have length num_layers - 1.
        """
        super(MultiLayerLinear, self).__init__()

        assert num_layers > 0, 'num_layers must be greater than 0'

        self.num_batch_norm_layers = num_layers
        self.num_linear_layers = num_layers
        self.num_dropout_layers = num_layers - 1

        if isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * num_layers
        if isinstance(batch_norm_eps, float):
            batch_norm_eps = [batch_norm_eps] * num_layers
        if isinstance(batch_norm_momentum, float):
            batch_norm_momentum = [batch_norm_momentum] * num_layers
        if isinstance(batch_norm_affine, bool):
            batch_norm_affine = [batch_norm_affine] * num_layers
        if isinstance(batch_norm_track_running_stats, bool):
            batch_norm_track_running_stats = [batch_norm_track_running_stats] * num_layers

        if isinstance(num_features, int):
            num_features = [num_features] * (num_layers + 1)
        if isinstance(bias, bool):
            bias = [bias] * num_layers

        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * (num_layers - 1)
        if isinstance(dropout_inplace, bool):
            dropout_inplace = [dropout_inplace] * (num_layers - 1)

        assert len(batch_norm) == num_layers, 'batch_norm must have length num_layers'
        assert len(batch_norm_eps) == num_layers, 'batch_norm_eps must have length num_layers'
        assert len(batch_norm_momentum) == num_layers, 'batch_norm_momentum must have length num_layers'
        assert len(batch_norm_affine) == num_layers, 'batch_norm_affine must have length num_layers'
        assert len(batch_norm_track_running_stats) == num_layers, 'batch_norm_track_running_stats must have length num_layers'

        assert len(num_features) == num_layers + 1, 'linear_num_features must have length num_layers + 1'
        assert len(bias) == num_layers, 'linear_bias must have length num_layers'

        assert len(dropout_p) == num_layers - 1, 'dropout_p must have length num_layers - 1'
        assert len(dropout_inplace) == num_layers, 'dropout_inplace must have length num_layers - 1'

        self.batch_norm = batch_norm
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_track_running_stats = batch_norm_track_running_stats

        self.linear_num_features = num_features
        self.linear_bias = bias

        self.dropout_p = dropout_p
        self.dropout_inplace = dropout_inplace

        self.modules = nn.Sequential()

        for i in range(num_layers):
            if batch_norm[i]:
                self.modules.append(nn.BatchNorm1d(num_features=num_features[i],
                                                   eps=batch_norm_eps[i],
                                                   momentum=batch_norm_momentum[i],
                                                   affine=batch_norm_affine[i],
                                                   track_running_stats=batch_norm_track_running_stats[i]))
            self.modules.append(nn.Linear(in_features=num_features[i],
                                          out_features=num_features[i + 1],
                                          bias=bias[i]))
            if i < num_layers - 1 and dropout_p[i] > 0:
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