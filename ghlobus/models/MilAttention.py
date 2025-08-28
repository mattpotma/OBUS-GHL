"""
MilAttention.py

A PyTorch nn.Module that implements attention-based multiple instance learning.

Based on the article:
    M Ilse, J Tomczak, M Welling, Attention-based deep multiple instance learning.
    arXiv:1802.04712v4 [cs.LG] 28 Jun 2018

And the Keras library
    https://keras.io/examples/vision/attention_mil_classification/

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import torch
from torch import nn
from typing import Tuple, Union, List
import torch.nn.functional as F


class MilAttention(nn.Module):
    """
    A PyTorch nn.Module that implements attention-based multiple instance learning.

    Args:
        input_dim: int          Dimension of the input tensor.
        embedding_dim: None, int, or List[int, int]
                                Specify up to two layers of projections of the input tensor.
                                Set to None to keep original dimension.
        attention_dim: int      Dimension of the intermediate attention tensor.
        use_gated: bool         Whether to use Gated Attention.
        dropout: float          between 0. and 1., fraction of units that drop out.

    Equations:
        attention weight a_k for the kth instance in a bag, h_k
            a_k = exp{w^T [tanh(V h_k^T) * sigmoid(U h_k^T)]}
        where w, V, and U are the weights of fully connected attention layers
        the attention-weighted context vector for the bag
            z = sum_k(a_k * h_k)

    Attributes:
        v_fc_layer: nn.Linear        FC layer that projects inputs to low-dimensional
                                       embedding (attention).
        u_fc_layer: nn.Linear        FC layer that projects inputs to low-dimensional
                                       embedding (gated attention).
        attn_fc_layer: nn.Linear     FC layer that projects the softmax logits to a
                                       scalar attention weights.
    """
    def __init__(self,
                 input_dim: int = 1000,
                 embedding_dim: Union[None, int, List[int]] = None,
                 attention_dim: int = 16,
                 use_gated: bool = True,
                 dropout: float = 0.5,
                 ) -> None:
        """
        Initializes the MilAttention module with configurable input, embedding,
        attention dimensions, gating, and dropout.

        Args:
            input_dim (int):                          Dimension of the input tensor.
            embedding_dim (None, int, or List[int]):  Specifies up to two layers of projections
                                                        for the input tensor. Set to None to keep
                                                        the original dimension.
            attention_dim (int):                      Dimension of the intermediate attention tensor.
            use_gated (bool):                         Whether to use gated attention mechanism.
            dropout (float):                          Fraction of units to drop out (between 0 and 1).

        Defines projection layers, attention layers, and optional gating and dropout layers for
        attention-based multiple instance learning.
        """
        super().__init__()

        # See if we are projecting input to embedding dimension
        if embedding_dim is None:
            # No projection
            self.q_fc_layer1 = None
            self.q_do_layer1 = None
            self.q_fc_layer2 = None
            self.q_do_layer2 = None
            final_embedding_dim = input_dim
        elif isinstance(embedding_dim, int):
            # Only one projection to new (lower?) dimension
            self.q_fc_layer1 = nn.Linear(input_dim, embedding_dim)
            self.q_do_layer1 = nn.Dropout(dropout)
            self.q_fc_layer2 = None
            self.q_do_layer2 = None
            final_embedding_dim = embedding_dim
        elif isinstance(embedding_dim, list) or isinstance(embedding_dim, tuple):
            # Check if it complies with expected one or two projections
            if len(embedding_dim) < 1 or len(embedding_dim) > 2:
                raise ValueError(f"embedding_dim must be a list of 1 or 2 elements, got {embedding_dim}.")
            # At least one projection to new (lower?) dimension
            self.q_fc_layer1 = nn.Linear(input_dim, embedding_dim[0])
            self.q_do_layer1 = nn.Dropout(dropout)
            # Check if there's a second projection
            if len(embedding_dim) == 2:
                # Second projection to a new (lower?) dimension
                self.q_fc_layer2 = nn.Linear(embedding_dim[0], embedding_dim[1])
                self.q_do_layer2 = nn.Dropout(dropout)
                final_embedding_dim = embedding_dim[1]
            else:
                # Even though a list was used, there's only one projection
                self.q_fc_layer2 = None
                self.q_do_layer2 = None
                final_embedding_dim = embedding_dim[0]
        else:
            # Invalid projection specification
            raise ValueError(f"Invalid embedding_dim {embedding_dim}")

        # "V" factor of MIL attention
        self.v_fc_layer = nn.Linear(in_features=final_embedding_dim,
                                    out_features=attention_dim)

        # Create gated attention factor?
        if use_gated:
            # "U" factor of gated MIL attention
            self.u_fc_layer = nn.Linear(in_features=final_embedding_dim,
                                        out_features=attention_dim)
        else:
            # No gating
            self.u_fc_layer = None

        # Projection to single dimension to obtain MIL attention weights
        self.attn_fc_layer = nn.Linear(in_features=attention_dim,
                                       out_features=1)

        # Final dropout layer after summing over instances of attention-weighted embeddings
        self.final_do_layer = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the attention weights and context vector.

        Args:
            inputs: Tensor              Input tensor.

        Returns:
            context_vector: Tensor      Context vector.
            attention_weights: Tensor   Attention weights.
        """
        # See if we are projecting inputs to new embedding dimension
        embeddings = inputs
        # Try each of two projection layers
        if self.q_fc_layer1 is not None:
            # First projection
            embeddings = F.relu(self.q_do_layer1(self.q_fc_layer1(embeddings)))
        if self.q_fc_layer2 is not None:
            # Second projection
            embeddings = F.relu(self.q_do_layer2(self.q_fc_layer2(embeddings)))

        # Compute attention
        attention = torch.tanh(self.v_fc_layer(embeddings))

        # Check if gating specified
        if self.u_fc_layer is not None:
            # Compute the gated attention factor
            gated_attention = torch.sigmoid(self.u_fc_layer(embeddings))
            # Compute pointwise vector product
            attention = attention * gated_attention

        # Compute the attention weights before softmax
        attention_logits = self.attn_fc_layer(attention)

        # Compute softmax normalized attention weights
        attention_weights = torch.softmax(attention_logits, dim=1)

        # Compute the bag context vector
        context_vector = torch.sum(attention_weights * embeddings, dim=1)
        context_vector = self.final_do_layer(context_vector)

        # Return the bag context vector and attention weights
        return context_vector, attention_weights
