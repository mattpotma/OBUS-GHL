"""
TvCnn.py

Generic TorchVision pre-trained CNN model.
Uses ImageNet 1k by default (for 1k output vector, which is assumed in other models).

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from torch import nn
from torchvision.models import get_model, get_weight


class TvCnn(nn.Module):
    def __init__(self,
                 tv_model_name: str = 'MobileNet_V2',
                 tv_weights_name: str = 'IMAGENET1K_V2',
                 ):
        """
        Generic TorchVision pre-trained CNN model. This is a convenience nn.Module definition that uses
        TorchVision pretrained models and retrieves them by name. Uses ImageNet 1k by default (for 1k output 
        vector, which is assumed in defaults of other models).

        Parameters:
            tv_model_name: str     - Name of the model to use in TorchVision
            tv_weights_name: str   - suffix/short version of the weights name to use
        """
        super(TvCnn, self).__init__()

        # Initialize the parameters
        self.tv_name = tv_model_name
        self.weight_name = "{}_Weights.{}".format(
            tv_model_name, tv_weights_name)

        # Retrieve the model and weights
        self.cnn = get_model(self.tv_name,
                             weights=self.weight_name)
        self.cnn_weights = get_weight(self.weight_name)

        # Define the required pre-processing:
        self.preprocess = self.cnn_weights.transforms(antialias=False)

    def forward(self, x):
        """
        Forward pass through the TorchVision CNN.

        Parameters:
            x: torch.Tensor   - data to pass through the model
        """
        # Check if the input is the correct shape
        reshaped = False
        if x.dim() == 5:
            # Assuming input is of shape (Batch, Frame, Channel, Height, Width)
            B, F, C, H, W = x.size()
            # Stack frames into a single batch dimension
            # This is necessary for the CNN to process the input correctly
            # Reshape to (Batch * Frame, Channel, Height, Width)
            x = x.view(B * F, C, H, W)
            reshaped = True
        # Preprocess the input data
        x = self.preprocess(x)

        # Process the input through the CNN
        x = self.cnn(x)

        # If reshaped, we need to reshape back to (Batch, Frame, FeatureVector)
        if reshaped:
            # Reshape back to (Batch, Frame, FeatureVector)
            x = x.view(B, F, -1)

        # Return feature vector from CNN
        return x
