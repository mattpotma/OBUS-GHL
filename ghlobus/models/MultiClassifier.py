"""
MultiClassifier.py

Multi-class classifier model that uses a single fully connected layer
to project the input features to the number of classes. Uses log(softmax)
activation, which is more numerically stable than softmax.

Author: Olivia Zahn

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from torch import nn
from torch.nn import functional as F


class MultiClassifier(nn.Module):
    """
    Multi-class classifier model using log(softmax) activation.
    Uses a single fully connected layer to project the input features
    to the number of classes.

    Args:
        in_features: int    Input feature vector length (input dimension).
        num_classes: int    Number of classes to classify (output dimension).
    """
    def __init__(self,
                 in_features: int = 1024,
                 num_classes: int = 2):
        # Invoke the super class' constructor
        super(MultiClassifier, self).__init__()
        # Create fully connected layer
        self.fc1 = nn.Linear(in_features, num_classes)
        # Store number of classes
        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass of the classifier model. Enforces batch dimension,
        if necessary.

        Parameters:
            x: torch.Tensor    Input feature tensor.

        Returns:
            torch.Tensor       Log(softmax) output of the classifier.
        """
        # Pass the embeddings through the FC classifier layer
        logits = self.fc1(x)
        # Add a leading dimension if batch_size == 1
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        # Compute the log_softmax for stability
        out = F.log_softmax(logits, dim=1)

        # Return both outputs and logits
        return out, logits
