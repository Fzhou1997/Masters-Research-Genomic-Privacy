from typing import Sequence

from torch import nn


class MultiLayerLinear(nn.Module):
    """
    A multi-layer linear module with optional batch normalization and dropout.

    Attributes:
        num_linear_layers (int): The number of linear layers in the module.
        num_inner_layers (int): The number of inner layers in the module.

        linear_num_features (Sequence[int]): A list of the number of features in each layer.
        linear_bias (Sequence[bool]): A list of booleans indicating whether to add a learnable bias to the output.

        batch_norm (Sequence[bool]): A list of booleans indicating whether to use batch normalization for each layer.
        batch_norm_eps (Sequence[float]): A list of epsilon values added to the denominator for numerical stability.
        batch_norm_momentum (Sequence[float]): A list of values used for the running_mean and running_var computation.
        batch_norm_affine (Sequence[bool]): A list of booleans indicating whether the module has learnable affine parameters.
        batch_norm_track_running_stats (Sequence[bool]): A list of booleans indicating whether the module tracks the running mean and variance.

        dropout_p (Sequence[float]): A list of dropout probabilities for each layer.
        dropout_inplace (Sequence[bool]): A list of booleans indicating whether to operate in-place.
        dropout_first (Sequence[bool]): If True, applies dropout before the batch normalization layer.

        modules (nn.Sequential): The sequential container for the module.
    """

    num_linear_layers: int
    num_inner_layers: int

    linear_num_features: Sequence[int]
    linear_bias: Sequence[bool]

    batch_norm: Sequence[bool]
    batch_norm_eps: Sequence[float]
    batch_norm_momentum: Sequence[float]
    batch_norm_affine: Sequence[bool]
    batch_norm_track_running_stats: Sequence[bool]

    dropout_p: Sequence[float]
    dropout_inplace: Sequence[bool]
    dropout_first: Sequence[bool]

    modules = nn.Sequential

    def __init__(self,
                 num_layers: int,
                 num_features: int | Sequence[int],
                 bias: bool | Sequence[bool],
                 batch_norm: bool | Sequence[bool],
                 batch_norm_eps: float | Sequence[float],
                 batch_norm_momentum: float | Sequence[float],
                 batch_norm_affine: bool | Sequence[bool],
                 batch_norm_track_running_stats: bool | Sequence[bool],
                 dropout_p: float | Sequence[float],
                 dropout_inplace: bool | Sequence[bool],
                 dropout_first: bool | Sequence[bool]):
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

        self.num_linear_layers = num_layers
        self.num_inner_layers = num_layers - 1

        if isinstance(num_features, int):
            num_features = [num_features] * (self.num_linear_layers + 1)
        if isinstance(bias, bool):
            bias = [bias] * self.num_linear_layers

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

        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * self.num_inner_layers
        if isinstance(dropout_inplace, bool):
            dropout_inplace = [dropout_inplace] * self.num_inner_layers
        if isinstance(dropout_first, bool):
            dropout_first = [dropout_first] * self.num_inner_layers

        assert len(num_features) == num_layers + 1, 'linear_num_features must have length num_layers + 1'
        assert len(bias) == num_layers, 'linear_bias must have length num_layers'

        assert len(batch_norm) == num_layers - 1, 'batch_norm must have length num_layers - 1'
        assert len(batch_norm_eps) == num_layers - 1, 'batch_norm_eps must have length num_layers - 1'
        assert len(batch_norm_momentum) == num_layers - 1, 'batch_norm_momentum must have length num_layers - 1'
        assert len(batch_norm_affine) == num_layers - 1, 'batch_norm_affine must have length num_layers - 1'
        assert len(batch_norm_track_running_stats) == num_layers - 1, 'batch_norm_track_running_stats must have length num_layers - 1'

        assert len(dropout_p) == num_layers - 1, 'dropout_p must have length num_layers - 1'
        assert len(dropout_inplace) == num_layers - 1, 'dropout_inplace must have length num_layers - 1'
        assert len(dropout_first) == num_layers - 1, 'dropout_first must have length num_layers - 1'

        self.linear_num_features = num_features
        self.linear_bias = bias

        self.batch_norm = batch_norm
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_track_running_stats = batch_norm_track_running_stats

        self.dropout_p = dropout_p
        self.dropout_inplace = dropout_inplace
        self.dropout_first = dropout_first

        self.modules = nn.Sequential()

        for i in range(num_layers):
            self.modules.append(nn.Linear(in_features=num_features[i],
                                          out_features=num_features[i + 1],
                                          bias=bias[i]))
            if i >= self.num_inner_layers:
                continue
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