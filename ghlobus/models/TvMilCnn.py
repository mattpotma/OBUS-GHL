"""
TvMilCnn.py

Generic TorchVision pre-trained CNN model, with hooks needed for MIL model.
Uses ImageNet 1k by default (for 1k output vector, which is assumed in
other models). Also allows setting custom weights from a checkpoint.
Applies CNN across all frames (exam-level) or across video (video-level)
instances in a bag.

Definitions
    (a) video-level MIL: input shape: (B, K, L, C, H, W)
        B - Batch size
        K - Bag size
        L - Number of frames in each video
        C - Number of channels
        H - Height of frame
        W - Width of frame
    (b) frame-level MIL: input shape (B, L, C, H, W)
        B - Batch size
        L - Bag size (number of frames in a bag, across videos)
        C - Number of channels
        H - Height of frame
        W - Width of frame

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import torch
from torch import nn
from torchvision.models import get_model, get_weight


class TvMilCnn(nn.Module):
    def __init__(self,
                 tv_model_name: str = 'MobileNet_V2',
                 tv_weights_name: str = 'IMAGENET1K_V2',
                 mil_format: str = 'frame',
                 pretrained_path: str = None,
                 pretrained_type: str = 'Cnn2RnnClassifier',
                 pretrained_freeze: bool = False,
                 ):
        """
        Generic TorchVision pre-trained CNN model. This is a convenience nn.Module
        definition that uses TorchVision pretrained models and retrieves them by name.
        Uses ImageNet 1k by default (for 1k output vector, which is assumed in defaults
        of other models). It also allows initialized with a checkpoint from a
        Cnn2RnnRegressor or Cnn2RnnClassifier model.

        Parameters:
            tv_model_name: str       - Name of the model to use in TorchVision
            tv_weights_name: str     - suffix/short version of the weights name to use
            mil_format: str          - Either 'frame' or 'video'
            pretrained_path: str     - path to the pretrained model checkpoint
            pretrained_type: str     - 'Cnn2RnnClassifier' or 'Cnn2RnnRegressor'
                                        or just plain 'Cnn'
            pretrained_freeze: bool  - whether to freeze the pretrained CNN weights
        """
        super(TvMilCnn, self).__init__()

        # Initialize the parameters
        self.tv_name = tv_model_name
        self.weight_name = f"{tv_model_name}_Weights.{tv_weights_name}"
        if mil_format not in ['frame', 'video']:
            raise ValueError(f"Unrecognized mil_format: {mil_format}")
        self.mil_format = mil_format

        # Retrieve the model and weights
        self.cnn = get_model(self.tv_name,
                             weights=self.weight_name)
        self.cnn_weights = get_weight(self.weight_name)

        # Load pretrained weights if specified
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path,
                                          pretrained_type,
                                          pretrained_freeze)

        # Define the required pre-processing:
        self.preprocess = self.cnn_weights.transforms(antialias=False)

    def load_pretrained_weights(self,
                                pretrained_path: str,
                                pretrained_type: str,
                                pretrained_freeze: bool,
                                ) -> None:
        """
        Load pretrained weights from a checkpoint.

        Parameters:
            pretrained_path: str     - path to the pretrained model checkpoint
            pretrained_type: str     - 'Cnn2RnnClassifier' or 'Cnn2RnnRegressor'
                                        or just plain 'Cnn'
            pretrained_freeze: bool  - whether to freeze the pretrained CNN weights
        """
        # Load the pretrained weights
        pretrained_params = torch.load(
            pretrained_path,
            map_location=torch.device('cpu'))['state_dict']
        model_params = self.state_dict()
        for key, val in pretrained_params.items():
            # See if it's a Cnn2RnnRegressor/Classifier
            if pretrained_type.lower() == 'cnn2rnnclassifier' or \
                    pretrained_type.lower() == 'cnn2rnnregressor':
                # Extract only the 'cnn' weights
                if key[:4] == 'cnn.':
                    # Take the remainder of the key name
                    key_cnn = key[4:]
                else:
                    continue
            elif pretrained_type.lower() == 'cnn':
                # If just a Cnn (e.g., Foundation model), use the key as is
                key_cnn = key
            else:
                raise ValueError(f"Unrecognized pretrained type: {pretrained_type}")
            # Check if the key is in the model parameters
            if key_cnn in model_params and val.shape == model_params[key_cnn].shape:
                model_params[key_cnn] = val.to(model_params[key_cnn].device)
                if pretrained_freeze:
                    # Freeze this layer of the model, if specified
                    model_params[key_cnn].requires_grad = False
        # Load the parameters in the model
        self.load_state_dict(model_params)

    # noinspection PyUnboundLocalVariable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TorchVision CNN.

        Parameters:
            x: torch.Tensor   - Batch of data to pass through the model
                                Two use cases: (a) video-level and (b) frame-level instances
                                (a) Shape: (B, K, L, C, H, W)
                                    B - Batch size
                                    K - Bag size
                                    L - Number of frames in each video
                                    C - Number of channels
                                    H - Height of frame
                                    W - Width of frame
                                (b) Shape (B, L, C, H, W)
                                    B - Batch size
                                    L - Bag size (number of frames in a bag, across videos)
                                    C - Number of channels
                                    H - Height of frame
                                    W - Width of frame
        Returns:
            torch.Tensor      - CNN embedding (feature vector)
                                (a) Shape: (B, K, L, FEATURE_DIM)
                                    B - Batch size
                                    K - Bag size
                                    L - Number of frames in each video
                                    FEATURE_DIM - Feature dimension of the CNN
                                (b) Shape: (B, L, FEATURE_DIM)
                                    B - Batch size
                                    L - Bag size (number of frames in a bag, across videos)
                                    FEATURE_DIM - Feature dimension of the CNN
        """
        # Based on mil_format, combine initial dimensions
        if self.mil_format == 'video' or self.mil_format == 'clip':
            # In video-MIL or clip-MIL, combine Batch, Bag, and Frames
            B, K, L, C, H, W = x.size()
            x = x.view(B * K * L, C, H, W)
        elif self.mil_format == 'frame':
            # In frame-MIL, combine Batch and Frames
            B, L, C, H, W = x.size()
            x = x.view(B * L, C, H, W)

        # Apply the typical preprocessing transform to the data
        x = self.preprocess(x)

        # Perform feature extraction
        z = self.cnn(x)

        # Re-expand initial dimensions, according to mil_format
        if self.mil_format == 'video' or self.mil_format == 'clip':
            # In video-MIL, revert Batch, Bag, and Frames
            z = z.view(B, K, L, -1)
        elif self.mil_format == 'frame':
            # In frame-MIL, revert Batch and Frames
            z = z.view(B, L, -1)

        # Return CNN embedding (feature vector)
        return z
