"""
SeparableConv2d.py

Utility class for a separable convolution layer. This is a depthwise convolution,
followed by a pointwise convolution across the input channels.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from torch import nn


class SeparableConv2d(nn.Module):
    """
    Separable convolution layer. This is a depthwise separable convolution,
    which performs a depthwise convolution followed by a pointwise convolution
    across the input channels.

    Parameters:
        in_channels: int     - Number of input channels
        out_channels: int    - Number of output channels
        kernel_size: int     - Size of the convolutional kernel
        padding: int         - Padding to apply to the input
        groups: int          - Number of groups to use in the convolution
        bias: bool           - Whether to use bias in the convolution
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 groups=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=kernel_size,
                                   groups=in_channels,
                                   padding=padding,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   groups=groups,
                                   bias=bias)

    def forward(self, x):
        """
        Forward pass of the separable convolution layer.

        Parameters:
            x: torch.Tensor  - Input tensor to the layer

        Returns:
            torch.Tensor     - Output tensor of the layer
        """
        temp = self.depthwise(x)
        out = self.pointwise(temp)
        return out
