"""
TvConvLSTM.py

Implementation of Convolutional LSTM in PyTorch.
Based on: https://github.com/ndrplz/ConvLSTM_pytorch
Added the following features to this ConvLSTM implementation
- Bottleneck convolutional layer for improved performance
  See https://arxiv.org/abs/1903.10172 for more details.
- Support for grouped convolutions for reduced computation.
  See https://arxiv.org/abs/1703.09938 for more details.
- For an illuminating introduction to LSTMs, see
  https://colah.github.io/posts/2015-08-Understanding-LSTMs/

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import torch
from torch import nn
from typing import Tuple, Union
from ghlobus.models.SeparableConv2d import SeparableConv2d


class ConvLSTMCell(nn.Module):
    """
    Core Convolutional LSTM cell. The wrapper will utilize
    a specified number of these cells as LSTM layers in the network.

    Definitions:
        B: Batch size
        L: Number of frames in the video
        C: Number of channels in the input video
        H: Height of the input video (rows)
        W: Width of the input video (columns)
        h: Hidden state
        c: Cell state
        b: Bottleneck state
        i: Input gate
        f: Forget gate
        o: Output gate
        g: Cell gate

    Parameters:
        input_size: int     Input tensor number of channels C (B, L, C, H, W).
        hidden_size: int    Hidden state number of channels D (B, L, D, H, W).
        kernel_size: int    Size of the convolutional kernel K (K, K).
        num_groups: int     Number of groups in which to segregate convolutions
        bias: bool          Whether to add bias. True or False
    """
    def __init__(self,
                 input_size: int = 1280,
                 hidden_size: Union[int, Tuple[int, int]] = 512,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 num_groups: int = 4,
                 bias: bool = True,
                 ):
        """
        Initialize ConvLSTMCell object.
        """
        # call parent constructor
        super(ConvLSTMCell, self).__init__()

        # save parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.padding = kernel_size // 2
        self.bias = bias

        # set up bottleneck convolutional layer
        self.bconv = SeparableConv2d(in_channels=self.input_size + self.hidden_size,
                                     out_channels=self.hidden_size,
                                     kernel_size=self.kernel_size,
                                     padding=self.padding,
                                     groups=self.num_groups,
                                     bias=self.bias)
        # set up convolution layers for input, forget, output, and gcell
        self.convs = SeparableConv2d(in_channels=self.hidden_size,
                                     out_channels=4 * self.hidden_size,
                                     kernel_size=self.kernel_size,
                                     padding=self.padding,
                                     groups=self.num_groups,
                                     bias=self.bias)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor],
                ):
        """
        Define forward pass.

        Parameters:
            inputs: Tensor      Current input video frame: (B, L=1, C, H, W).
            states: tuple       Tuple of previous hidden_state, cell_state
                                (h_prev, c_prev)

        Returns:
            state_next: tuple   Tuple of next hidden_state, cell_state, bottleneck_state
                                (h_next, c_next, b_next)
        """
        # split states into hidden, cell
        h_prev, c_prev = states

        # concatenate input and hidden along channel axis
        concat = torch.cat([inputs, h_prev], dim=1)

        # bottleneck convolution M + N --> N
        b_next = torch.relu(self.bconv(concat))

        # remaining LSTM convolutions
        all_convs = self.convs(b_next)

        # split out LSTM component convolution results
        ic, fc, oc, gc = torch.split(all_convs, self.hidden_size, dim=1)

        # apply activations
        i_next = torch.sigmoid(ic)
        f_next = torch.sigmoid(fc)
        o_next = torch.sigmoid(oc)
        g_next = torch.relu(gc)

        # generate next hidden, cell states
        c_next = f_next * c_prev + i_next * g_next
        h_next = o_next * torch.relu(c_next)

        return h_next, c_next, b_next

    def init_states(self, batch_size: int, image_size: Tuple[int, int]):
        """
        Defines initial hidden and cell states for the LSTM cell.
        These start out as all zeros and evolve during training.

        Parameters:
            batch_size:  int                Number of samples in the batch
            image_size:  Tuple[int, int]    Size of the input image (H, W)
        """
        height, width = image_size
        device = self.convs.depthwise.weight.get_device()
        if isinstance(device, int) and device < 0:
            device = 'cpu'
        initial_hidden = initial_cell = torch.zeros(batch_size,
                                                    self.hidden_size,
                                                    height,
                                                    width,
                                                    device=device)
        return initial_hidden, initial_cell


class TvConvLSTM(nn.Module):
    """
    Wrapper for ConvLSTMCell to create a multi-layer ConvLSTM network.

    Definitions:
        B: Batch size
        L: Number of frames in the video
        C: Number of channels in the input video
        H: Height of the input video (rows)
        W: Width of the input video (columns)
        h: Hidden state
        c: Cell state
        b: Bottleneck state

    Parameters:
        input_size:  int      Number of channels in input
        hidden_size: int      Number of hidden channels
        kernel_size: int      Size of kernel in LSTM convolutions
        num_layers: int       Number of LSTM layers stacked on each other
        num_groups: int       Number of groups in which to segregate convolutions
        batch_first: bool     Whether dimension 0 or 1 is the batch
        bias: bool            Whether to include bias in Convolution

    Input:
        A tensor of size (B, L, C, H, W) or (L, B, C, H, W)
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length L of each output
            1 - last_state_list is the list of last states
                each element of the list is a tuple (h, c)
                for hidden state and cell state
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = TvConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
    def __init__(self,
                 input_size: int = 1280,
                 hidden_size: Union[int, Tuple[int, int]] = 512,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 num_layers: int = 1,
                 num_groups: int = 4,
                 batch_first: bool = True,
                 bias: bool = True,
                 ):
        super(TvConvLSTM, self).__init__()

        # Ensure that both `kernel_size` and `hidden_size` are lists having len == num_layers
        if isinstance(kernel_size, list):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        if isinstance(hidden_size, list):
            self.hidden_size = hidden_size
        else:
            self.hidden_size = self._extend_for_multilayer(hidden_size, num_layers)

        # store other attributes
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.batch_first = batch_first
        self.bias = bias

        # put LSTM cell blocks in a list
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_size = self.input_size if i == 0 else self.hidden_size[i - 1]

            cell_list.append(ConvLSTMCell(input_size=cur_input_size,
                                          hidden_size=self.hidden_size[i],
                                          kernel_size=self.kernel_size[i],
                                          num_groups=self.num_groups,
                                          bias=self.bias))
        # convert to ModuleList
        self.cell_list = nn.ModuleList(cell_list)

        # pool spatial layers
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, inputs):
        """
        Parameters:
            inputs: Tensor     Input 5-D Tensor
                               with shape (B, L, C, H, W) or (L, B, C, H, W)

        Returns:
            Tuple of Tensors   last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            inputs = inputs.permute(1, 0, 2, 3, 4)

        # determine batch size and image size
        B, L, _, H, W = inputs.size()

        # Since the init is done in forward. Can send image size here
        states = self._init_states(batch_size=B,
                                   image_size=(H, W))

        # grab the input
        cur_layer_input = inputs

        # iterate through layers of ConvLSTM blocks
        for layer_idx in range(self.num_layers):

            # grab the hidden and cell states for this layer
            h, c = states[layer_idx]

            # recurrent computation of ConvLSTM through time steps
            output_inner = []
            for t in range(L):
                # execute the ConvLSTM cell for each layer
                # and grab the hidden, cell, and bottleneck states
                h, c, b = self.cell_list[layer_idx](inputs=cur_layer_input[:, t, :, :, :],
                                                    states=[h, c])
                # keep track of hidden states through all time steps
                output_inner.append(h)

            # store the hidden states for this layer
            layer_output = torch.stack(output_inner, dim=1)
            # route the output of this layer to the next layer
            cur_layer_input = layer_output

        # Grab the final LSTM layer's last time step output (B, L=-1, C, H, W)
        # Concatenate it with the final LSTM layer's last time step bottleneck output
        # noinspection PyUnboundLocalVariable
        last_output = torch.cat([layer_output[:, -1, ...], b], dim=1)

        # Do spatial pooling on the last time step to generate ConvLSTM output
        pooled_output = self.pool(last_output).squeeze()

        # return both the pooled output of the last time step
        # and the output of all layers, which is infrequently used
        return pooled_output, layer_output

    def _init_states(self, batch_size, image_size):
        """
        Defines initial hidden and cell states for every LSTM layer.
        These start out as all zeros and evolve during training.

        Parameters:
            batch_size:  int                Number of samples in the batch
            image_size:  Tuple[int, int]    Size of the input image (H, W)

        Returns:
            init_states:  list              List of tuples of hidden, cell states
                                            for each layer
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_states(batch_size, image_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Expands a single parameter to a list of length num_layers.

        Parameters:
            param:  int or list    Single parameter or list of parameters
            num_layers: int        Number of layers in the network

        Returns:
            list of int            List of parameters of length num_layers
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

