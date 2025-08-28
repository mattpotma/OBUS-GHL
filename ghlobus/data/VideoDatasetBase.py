"""
VideoDatasetBase.py

Base class for video datasets.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
from abc import ABC, abstractmethod
from time import sleep
from typing import Tuple, Any, List, Union, Callable, Literal

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDatasetBase(Dataset, ABC):
    """
    Abstract base class for video datasets.

    This class provides the core functionality for loading and preprocessing data
    from video datasets. Subclasses are expected to implement the abstract
    methods to handle specific data processing and batch creation logic.

    Parameters:
        df (pd.DataFrame):                 DataFrame containing the dataset metadata.
        transforms (tuple):                Tuple of functions to transform the data.
        channels (int):                    Number of channels for the output tensor
                                           (1 for mono, 3 for RGB).
        path_col (str):                    Column in the DataFrame that contains the file paths.
        label_cols (List[str]):            Column(s) in the DataFrame that contains the labels.
        data_dir (str, optional):          Base directory where the data is stored. If provided,
                                           it will be prepended to the file paths. Defaults to None.
        frames_or_channel_first :          Specifies whether the output tensor should have
            (Literal['frame', 'channel'])  frames first or channels first. Defaults to 'frame'.
        crop_fn (Callable, optional):      Function for frame cropping.
                                           Defaults to None.
        subsample (Callable, optional):    Function for frame subsampling.
                                           Defaults to None.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 transforms: tuple,
                 channels: int,
                 path_col: str,
                 label_cols: Union[str, List[str]],
                 data_dir: str = None,
                 frames_or_channel_first: Literal['frame',
                                                  'channel'] = 'frame',
                 crop_fn: Callable = None,
                 subsample: Callable = None):

        self.df = df
        self.transforms = list(transforms) if transforms else []
        self.channels = channels
        self.path_col = path_col
        self.label_cols = [label_cols] if isinstance(
            label_cols, str) else label_cols
        self.data_dir = data_dir
        self.frames_or_channel_first = frames_or_channel_first
        self.crop_fn = crop_fn
        self.subsample = subsample

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.df)

    def _get_path(self, idx: int) -> str:
        """
        Constructs the full path for a sample at a given index.

        Args:
            idx (int):     Index of the sample.

        Returns:
            str:           Path to the sample file.
        """
        # Get the path from the DataFrame path_column
        path = self.df.iloc[idx][self.path_col]
        if self.data_dir and not os.path.isabs(path):
            # Use os.path.join for platform-independent path construction
            path = os.path.join(self.data_dir, path)

        # Check if the path has a valid extension
        extensions_to_check = ['.mp4', '.dcm', '.pt']
        if not any(path.endswith(ext) for ext in extensions_to_check):
            # Default to .pt if no extension matches
            path += '.pt'

        return path

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Retrieves a sample from the dataset.

        This method orchestrates the loading, processing, and batching of a single
        data sample. It relies on abstract methods that must be implemented by
        subclasses.

        Args:
            idx (int):         Index of the sample to retrieve.

        Returns:
            A tuple containing the processed data and its corresponding label.
        """
        # Load the sample from the file path
        path = self._get_path(idx)
        frames = self.load_sample(path)

        # Apply common frame and label processing steps
        frames = self._common_frames_processing(frames)
        labels = self._common_labels_processing(idx)

        # Return the final batch, with logic specific to the dataset type
        # contained in `_get_batch`
        return self._get_batch(frames, labels)

    def _common_frames_processing(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies common preprocessing steps to the frames.

        This method can be overridden by subclasses to include additional
        preprocessing logic that is common across different dataset types.

        Args:
            frames (torch.Tensor):   Input tensor of frames.

        Returns:
            torch.Tensor:            Processed tensor of frames.
        """
        # STEP 1. Check for correct number of channels
        if self.channels == 3 and frames.ndim > 2 and frames.shape[1] == 1:
            # Repeat monochrome image across three channels
            frames = torch.repeat_interleave(frames, repeats=3, dim=1)

        # STEP 2. If a crop function is provided, apply it
        if self.crop_fn:
            frames = self.crop_fn(frames)

        # STEP 3. If subsampling is defined, apply it
        if self.subsample and callable(self.subsample):
            frames = self.subsample(frames)

        # STEP 4. Apply Dataset-specific processing to the frames
        frames = self._process_frames(frames)

        # STEP 5. Apply transformations to the frames
        frames = self.apply_transforms(frames)

        # Check the shape of the frames; move channel dimension if needed
        if self.frames_or_channel_first == 'channel':
            frames = torch.Tensor.movedim(frames, 1, 0)

        # Return the processed frames
        return frames

    def _common_labels_processing(self, idx: int) -> torch.Tensor:
        """
        Applies common preprocessing steps to the labels.

        This method can be overridden by subclasses to include additional
        preprocessing logic that is common across different dataset types.

        Args:
            idx (int): The index of the sample.

        Returns:
            torch.Tensor: The processed labels tensor.
        """
        labels = self.df.iloc[idx][self.label_cols].values
        labels = labels.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        return labels.squeeze()  # Ensure labels are 1D for this batch element :)

    @abstractmethod
    def _process_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        An abstract method for subclass-specific frame processing.

        This method should contain logic for operations like cropping, subsampling,
        or other transformations that are specific to the training or inference context.

        Args:
            frames (torch.Tensor): The input tensor of frames.

        Returns:
            torch.Tensor: The processed tensor of frames.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_batch(self, frames: torch.Tensor, idx: int) -> Tuple[Any, Any]:
        """
        An abstract method for creating the final batch.

        This method should handle the final arrangement of the data and label
        into a tuple for the DataLoader.

        Args:
            frames (torch.Tensor): The processed tensor of frames.
            idx (int): The index of the sample.

        Returns:
            A tuple containing the final data and label.
        """
        raise NotImplementedError

    @staticmethod
    def load_sample(filepath: str) -> torch.Tensor:
        """
        Loads a sample from a file path with a retry mechanism.

        Args:
            filepath (str): The path to the file.

        Returns:
            torch.Tensor: The loaded data as a tensor.

        Raises:
            Exception: If the file cannot be loaded after 10 attempts.
        """
        frames = None
        exception = None

        # Try loading the frames up to ten times
        for _ in range(10):
            try:
                frames = torch.load(filepath)
                break
            except Exception as err:
                # Store the given error
                exception = err
                # Wait 1 second, then try loading again
                sleep(1)

        # If frames is still None, raise an exception
        if frames is None:
            raise Exception(
                f"Loading file {filepath} failed. Last exception raised:\n{exception}")

        return frames

    def apply_transforms(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply the transforms in self.transforms to the input frames.

        Parameters:
            frames (torch.Tensor): data Tensor containing frame data.

        Returns:
            torch.Tensor: data modified by the self.transforms functions.
        """
        for transform in self.transforms:
            frames = transform(frames)
        return frames
