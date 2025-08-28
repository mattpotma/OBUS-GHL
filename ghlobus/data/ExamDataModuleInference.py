"""
ExamInferenceDataModule.py

This Dataset module is used to dish out samples for inference
at an exam level (not video level). This is appropriate for models that need
exam level information during training. For example, models that use multiple
instance learning.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
# Python standard library
from typing import Any, Literal

# External imports
from torch.utils.data import DataLoader

# Internal imports
from ghlobus.data.ExamDataModuleBase import ExamDataModuleBase


class ExamDataModuleInference(ExamDataModuleBase):
    """
    A DataModule for video inference, supporting TWIN model.
    """
    def __init__(self,
                 dataset_name: Literal["train", "val", "test"],
                 **kwargs: Any,
                 ) -> None:
        """
        Initializes the ExamDataModuleInference.

        Parameters:
            dataset_name: Literal         - Name of the dataset to use ("train", "val", or "test").
            dataset_dir: os.PathLike      - Root directory containing data split definition files
            distribution: int             - Integer indicating the distribution ID to use
            transforms: List[Callable]    - List of functions to transform the data prior
                                            to getting passed to the model
            augmentations: List[Callable] - List of functions to transform the data prior
                                            to getting passed to the model, train time only
            batch_size: int               - Size of batch for dataloaders
            num_workers: int              - Number of workers for dataloaders
            bag_size: int                 - Size of bags for dataloaders
            mil_format: str               - 'video' or 'clip' or 'frame' ('clip' not yet supported)
            channels: int                 - Number of channels for output (keep 1 or repeat to 3)
            frames: int                   - Number of sample frames in each video (mil_format='video')
            frame_sampling: str           - Either 'random', 'jitter', 'uniform', or 'matern'
            frames_or_channel_first: str  - Either 'frames' or 'channel'
            use_known_combos: bool        - Whether to filter sweeps by KNOWN_COMBOS (**FIXED**)
            random_known_combos: bool     - Whether to randomly sample from KNOWN_COMBOS (**ON-THE-FLY**)
            allow_biometric: bool         - Whether to use biometric sweeps when insufficient KNOWN_COMBOS
            strict_known_combos: bool     - Whether to use strict definition of KNOWN_COMBOS
            upsample:                     - Whether to upsample exams from the minority class
            max_replicates: int           - Maximum number of times to replicate an exam
            balance_ga: bool              - Whether to balance GA when upsampling exams
            image_dims: int or (int,int)  - Size of output image for inference. CenterCrop
                                            is performed on the data to the given image dims.
        """
        super().__init__(**kwargs)
        # Record the dataset name
        self.dataset_name = dataset_name
        # Create the allowed stages
        self.allowed_stages = {
            'predict': [dataset_name],
        }


    def predict_dataloader(self) -> DataLoader:
        """
        Set up the prediction DataLoader.

        Returns
        ----------
            dataloader: DataLoader   (train, val, or test) prediction DataLoader.
        """
        return self._create_inference_dataloader(dataset_name=self.dataset_name)
