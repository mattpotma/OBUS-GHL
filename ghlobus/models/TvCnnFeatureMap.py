"""
TvCnnFeatureMap.py

Generic TorchVision pre-trained CNN model.
Uses ImageNet 1k pre-training by default.
Allows specification of output layer (including intermediate feature map layers).

Definitions:
    B   - batch_size
    L   - number of frames in video
    C   - input image channel count
    H   - input image height
    W   - input image width
    CFM - channels of the CNN output feature map
          when CNN is cut at the feature map layer
    HFM - height feature dimension
          when CNN is cut at the feature map layer
    WFM - width feature dimension
          when CNN is cut at the feature map layer

Author: Courosh Mehanian
        Olivia Zahn

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from typing import Union
from torch import nn

from ghlobus.models.TvCnn import TvCnn


class TvCnnFeatureMap(TvCnn):
    def __init__(self,
                 cnn_name: str = "MobileNet_V2",
                 cnn_weights_name: str = "DEFAULT",
                 cnn_layer_id: Union[str, int, None] = None,
                 ):
        """
        Generic TorchVision pretrained CNN model. This is a convenience nn.Module
        definition that uses TorchVision pretrained models and retrieves them by
        name. Uses ImageNet 1k weights by default, but can output an arbitrary
        feature map layer.

        Parameters:
            cnn_name: str           - Name of the model to use in TorchVision
            cnn_weights_name: str   - suffix/short version of the weights name to use
            cnn_layer_id: str, int  - Name or index of the feature map output layer
        """
        super(TvCnnFeatureMap, self).__init__(tv_model_name=cnn_name,
                                              tv_weights_name=cnn_weights_name)

        # Initialize the parameters
        self.cnn_name = self.tv_name
        self.cnn_weights_name = self.weight_name

        # define the feature layer output
        self.cnn_layer_id = cnn_layer_id
        self.feature_map = None

        # trim to feature map layer (modifies self.cnn directly)
        self._trim_to_layer()

    def _trim_to_layer(self):
        """
        Extract the convolutional layers up to the desired layer.
        """
        # Extract the convolutional layers up to the desired layer index
        if ('mobilenet' in self.cnn_name.lower() or
            'efficientnet' in self.cnn_name.lower()):
            # If the layer ID is not provided, use the entire feature extractor.
            # For MobileNet and EfficientNet this is accessed using model.features
            if self.cnn_layer_id is None:
                self.cnn = self.cnn.features
            else:
                if isinstance(self.cnn_layer_id, str):
                    layer_idx = int(self.cnn_layer_id)
                else:
                    layer_idx = self.cnn_layer_id
                if not len(self.cnn.features) > layer_idx:
                    raise ValueError(f"CNN {self.cnn_name} has no feature map layer {layer_idx}")
                self.cnn = self.cnn.features[0:layer_idx+1]
        elif 'resnet' in self.cnn_name.lower():
            # If the layer ID is not provided, use the entire feature extractor.
            # For all ResNets this is all but the last two layers of the network.
            if self.cnn_layer_id is None:
                self.cnn = nn.Sequential(*(list(self.cnn.children())[:-2]))
            else:
                if isinstance(self.cnn_layer_id, str):
                    layer_idx = int(self.cnn_layer_id)
                else:
                    layer_idx = self.cnn_layer_id
                self.cnn_features = nn.Sequential(*(list(self.cnn.children())[:-2]))
                if not len(self.cnn_features) > layer_idx:
                    raise ValueError(f"CNN {self.cnn_name} has no feature map layer {layer_idx}")
                self.cnn = self.cnn_features[0:layer_idx+1]
        else:
            raise ValueError(f"CNN {self.cnn_name} not yet implemented")

    def forward(self, x):
        """
        Forward pass through the TorchVision CNN.

        Parameters:
            x: torch.Tensor  - data to pass through the model

        Returns:
            torch.Tensor     - output of the model
        """
        # get dimensionality of input
        # assumes batch of multichannel videos
        B, L, C, H, W = x.size()

        # concatenate batch and frames for speed
        x = x.view(B * L, C, H, W)

        # Apply the typical preprocessing transform to the data
        x = self.preprocess(x)

        # evaluate the CNN
        z = self.cnn(x)

        # get dimensionality of output
        # CFM, HFM, WFM are channels, height, width of feature map
        _, CFM, HFM, WFM = z.size()

        # de-concatenate batch and frames
        z = z.view(B, L, CFM, HFM, WFM)

        # return feature map from intermediate feature map layer
        return z
