"""
VideoDataModuleTraining.py

This module is designed for training and validation, with options for data augmentation
and stratified sampling to ensure balanced and robust model training.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from typing import Callable, List

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import CenterCrop

from ghlobus.data.VideoDataModuleBase import VideoDataModuleBase
from ghlobus.data.VideoDatasetTraining import VideoDatasetTraining
from ghlobus.utilities.sample_utils import compute_equal_sampling_weights_by_trimester


class VideoDataModuleTraining(VideoDataModuleBase):
    """
    A DataModule for training and validating video models.
    """

    def __init__(self,
                 frames: int = 50,
                 augmentations: List[Callable] = [],
                 use_stratified_sampler: bool = True,
                 use_inference_val_dataset: bool = False,
                 **kwargs
                 ):
        """
        Initializes the VideoDataModuleTraining.

        Parameters
        ----------
            channels: int                       Number of channels for the output
                                                (1 for grayscale, 3 for RGB color).
            frames: int                         Number of frames to sample from each video.
            transforms: List[Callable]          List of functions for data transformation.
            augmentations: List[Callable]       List of functions for data augmentation,
                                                applied only during training.
            subsample: Callable, optional       Function for subsampling frames from a video.
            frames_or_channel_first: str        Specifies the dimension order
                                                ('frames' or 'channel').
            use_stratified_sampler: bool        If True, uses a stratified sampler for the
                                                training set.
            use_inference_val_dataset: bool     If True, uses the inference dataset for
                                                validation.
            label_cols: Union[str, List[str]]   Name of the column(s) in the DataFrame that
                                                contains the labels.
            path_col: str                       Column in DataFrame that contains the file paths.
            **kwargs:                           Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)
        # Set attributes
        self.frames = frames
        self.augmentations = augmentations
        # Set boolean attribute for sampling strategy
        self.use_stratified_sampler = use_stratified_sampler
        self.use_inference_val_dataset = use_inference_val_dataset
        # Initialize the allowed_stages and corresponding datasets
        self.allowed_stages = {
            'fit': ['train', 'val'],
        }
        # Set the crop function
        self._crop_fn = CenterCrop(self.image_dims)

    def _create_train_dataloader(self, dataset_name: str) -> DataLoader:
        """
        Creates a DataLoader for training.
        """
        # Set up the transforms, and add augmentations for training
        transforms = self.transforms.copy()
        if dataset_name == 'train':
            transforms += self.augmentations

        # Create the dataset for training
        df = self.dfs[dataset_name]
        dataset = VideoDatasetTraining(
            df=df,
            path_col=self.path_col,
            label_cols=self.label_cols,
            transforms=transforms,
            channels=self.channels,
            data_dir=self.data_dir,
            frames_or_channel_first=self.frames_or_channel_first,
            crop_fn=self.crop_fn,
            subsample=self.subsample,
            frames=self.frames,
        )

        # Set up the Sampler, if applicable
        sampler = None
        if self.use_stratified_sampler and dataset_name == 'train':
            weights = compute_equal_sampling_weights_by_trimester(df)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.DoubleTensor(weights),
                num_samples=len(weights)
            )

        # Set the DataLoader
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training set.
        """
        return self._create_train_dataloader(dataset_name='train')

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation set.
        """
        if self.use_inference_val_dataset:
            return self._create_inference_dataloader(dataset_name='val')
        return self._create_train_dataloader(dataset_name='val')

    @property
    def crop_fn(self) -> Callable:
        """
        Returns the crop function for the dataset.
        """
        return self._crop_fn
