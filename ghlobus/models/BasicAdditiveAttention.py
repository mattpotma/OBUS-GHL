"""
BasicAdditiveAttention.py

A PyTorch nn.Module that implements a basic additive attention scheme.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import torch
from typing import Tuple
from torch import nn


class BasicAdditiveAttention(nn.Module):
    """
    A PyTorch nn.Module that implements a basic additive attention scheme.

    Args:
        input_dim (int):                   Dimension of the input tensor.
        attention_dim (int):               Dimension of the attention tensor.

    Attributes:
        linear_in (nn.Linear):             FC layer to transform the input tensor.
        linear_out (nn.Linear):            FC layer to transform the attention tensor.

    Methods:
        forward(inputs):                   Computes the attention weights and context vector.

    Returns:
        context_vector (torch.Tensor):     Context vector.
        attention_weights (torch.Tensor):  Attention weights.
    """
    
    def __init__(self, input_dim: int = 1000, attention_dim: int = 16) -> None:
        """
        Initializes the input and attention dimensions, and defines two linear layers.

        Args:
            input_dim (int):               Dimension of the input tensor.
            attention_dim (int):           Dimension of the attention tensor.
        """
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim

        # Define linear attention layers
        self.linear_in = nn.Linear(in_features=input_dim, 
                                   out_features=attention_dim)
        self.linear_out = nn.Linear(in_features=attention_dim, 
                                    out_features=1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the attention weights and context vector.

        Args:
            inputs (torch.Tensor):             Input tensor.

        Returns:
            context_vector (torch.Tensor):     Context vector.
            attention_weights (torch.Tensor):  Attention weights.
        """
        # Compute the attention weights
        attention_weights = self.linear_in(inputs)
        attention_weights = torch.tanh(attention_weights)
        attention_weights = self.linear_out(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Compute the context vector
        context_vector = attention_weights * inputs
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights


class MultipleAdditiveAttention(nn.Module):
    """
    A PyTorch nn.Module that implements a module with multiple additive attention.

    Args:
        input_dim: int              Dimension of the input tensor.
        attention_dim: int          Dimension of the attention tensor.
        num_modules: int            Number of attention modules to use.

    Attributes:
        attention_modules: nn.ModuleList   List of BasicAdditiveAttention modules.
    """
    def __init__(self, input_dim: int = 1000, attention_dim: int = 16, num_modules: int = 8) -> None:
        """
        Defines multiple attention modules.
        Initializes the input and attention dimensions, and defines number of modules.

        Args:
            input_dim (int):        Dimension of the input tensor.
            attention_dim (int):    Dimension of the attention tensor.
            num_modules (int) :     Number of modules for multiple additive attention
        """
        super().__init__()
        # Define multiple attention modules using list comprehension
        self.attention_modules = nn.ModuleList([BasicAdditiveAttention(input_dim, attention_dim) for _ in range(num_modules)])
        self.num_modules = num_modules

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the attention weights and context vector.

        Args:
            inputs: Tensor              Input tensor.

        Returns:
            context_vector: Tensor      Context vector.
            attention_weights: Tensor   Attention weights.
        """
        context_vectors = []
        attention_weights = []
        for module in self.attention_modules:
            context_vector, attention_weight = module(inputs)
            context_vectors.append(context_vector)
            attention_weights.append(attention_weight)

        # Concatenate the context vectors from all attention modules
        # (B, input_dim) * num_modules -> (B, input_dim * num_modules)
        context_vectors = torch.cat(context_vectors, dim=1) / self.num_modules

        try:
            attention_weights = torch.Tensor(torch.vstack(attention_weights))
        except Exception as e:
            print(f"Error stacking attention weights: {e}")
            print(f"Attention weights: {attention_weights}")

        return context_vectors, attention_weights
