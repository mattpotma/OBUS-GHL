"""
VideoDataModuleBase.py

This module defines the VideoDataModuleBase, a foundational class for handling
video-based datasets in PyTorch Lightning. It encapsulates common logic for data loading,
initialization, and setup, which is then extended by specialized DataModules for
training and inference.

This base class is designed to be subclassed and is not intended for direct use.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
from typing import Union, List, Callable, Tuple

import pandas as pd
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from lightning import LightningDataModule

from ghlobus.utilities.paths import v4_dataset_dir
from ghlobus.utilities.constants import DIST_DIR
from ghlobus.utilities.inference_utils import LGA_MEAN, LGA_STD
from ghlobus.data.VideoDatasetInference import VideoDatasetInference

class VideoDataModuleBase(LightningDataModule):
    """
    A base class for Video-based DataModules, providing common functionality.
    """

    def __init__(self,
                 dataset_dir: os.PathLike = v4_dataset_dir,
                 batch_size: int = 8,
                 num_workers: int = 0,
                 distribution: Union[None, str, int] = 0,
                 data_file_template: str = '{}.csv',
                 label_cols: Union[str, List[str]] = 'z_log_ga',
                 path_col: str = 'outpath',
                 channels: int = 3,
                 image_dims: Union[int, Tuple[int, int]] = (256, 256),
                 frames_or_channel_first: str = 'frames',
                 transforms: List[Callable] = [],
                 filter_ga: float = None,
                 filter_subset: Union[None, str] = None,
                 subsample: Union[None, Callable[[
                     Tensor, int], Tensor]] = None,
                 ):
        """
        Initializes the VideoDataModuleBase.

        Args:
            dataset_dir (os.PathLike):          Root directory containing the dataset.
            batch_size (int):                   Size of each batch for the DataLoader.
            num_workers (int):                  Number of worker processes for data loading.
            distribution (None, str, int):      Distribution ID to use. If None, the root
                                                  dataset directory is used.
            data_file_template (str):           Template string for dataset CSV filenames.
            label_cols (str, List[str]):        Column(s) containing labels in the dataset.
            path_col (str):                     Column name for file paths in the dataset.
            channels (int):                     Number of image channels (e.g., 1 for grayscale, 3 for RGB).
            image_dims (int, Tuple[int, int]):  Image dimensions (height, width) for cropping.
            frames_or_channel_first (str):      Ordering of frames/channels ('frames' or 'channel').
            transforms (List[Callable]):        List of transformation functions to apply to the data.
            filter_ga (float, optional):        Threshold for filtering by gestational age (GA).
            filter_subset (None, str):          Optional name of a subset to filter on
                                                  (e.g., main_test, novice_test, ivf_test). This
                                                  column must exist as a binary column in the dataset.
            subsample (None, Callable):         Optional subsampling function for frames.
        """
        # Call the parent constructor
        super().__init__()
        self.base_dataset_dir = dataset_dir

        # See if distribution subfolder is specified
        if distribution is None:
            self.data_dir = dataset_dir
        else:
            self.data_dir = os.path.join(
                dataset_dir, DIST_DIR, str(distribution)
            )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_file_template = data_file_template
        self.dfs = dict()

        # Log the label columns and path columns
        self.label_cols = label_cols if isinstance(
            label_cols, list) else [label_cols]
        self.path_col = path_col

        # Set the number of channels, image dimensions, and frame order
        self.channels = channels
        self.image_dims = image_dims
        if isinstance(self.image_dims, int):
            self.image_dims = (self.image_dims, self.image_dims)
        self.frames_or_channel_first = frames_or_channel_first
        # Set the filter value, in days, for GA thresholding
        # (filtering removes GA below set value)
        self.filter_ga = filter_ga
        # Filtering based on test subset, if specified
        self.filter_subset = filter_subset

        # Set the transforms to be applied to the data
        self.transforms = transforms

        # Set the subsample function attribute
        self.subsample = subsample

        # Initialize the `allowed_stages` attribute
        self.allowed_stages = {}

    @property
    def crop_fn(self) -> Callable:
        """
        Returns a function for cropping video frames.

        Returns
        -------
            Callable[[torch.Tensor, int], torch.Tensor]
                A function that takes a tensor of video frames and the desired
                crop size, and returns the cropped frames.
        """
        raise NotImplementedError(
            "The crop_fn property must be implemented in subclasses."
        )

    def setup(self, stage: str = None):
        """
        Loads and prepares the data for inference.
        """
        # Check for valid stage
        if stage not in self.allowed_stages:
            raise ValueError(
                f"{self.__class__.__name__} can only be set up for {list(self.allowed_stages.keys())} stage.")

        # Gather the datasets to load based on the stage, as defined in the children classes
        datasets = self.allowed_stages[stage]
        if not datasets:
            raise ValueError(
                f"No datasets defined for stage '{stage}' in {self.__class__.__name__}.")

        # Load the one dataset CSV file per dataset
        self.dfs = {ds: self.load_dataset_csv(ds) for ds in datasets}

        # Ensure any column that can be converted to floats are converted, if they can be converted
        for df in self.dfs.values():
            for col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    print(f"Column {col} could not be converted to float. Skipping conversion.")

        # If the feature is `z_log_ga`, ensure to log-normalize the 'log_ga' values
        if 'z_log_ga' in self.label_cols:
            self._normalize_log_ga_boe()

        # Apply GA filter, if specified
        if self.filter_ga is not None:
            self._apply_ga_filter()

        # Apply subset filter, if specified
        if self.filter_subset is not None:
            self._apply_subset_filter()

    def load_dataset_csv(self, dataset_name: str) -> pd.DataFrame:
        """
        Loads a dataset from a CSV file.
        Parameters
        ----------

            dataset_name: str     Name of the dataset to load.

        Returns
        ----------
            pd.DataFrame          DataFrame containing the dataset.
        """
        filename = self.data_file_template.format(dataset_name)
        csv_path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(csv_path, dtype=str)
        # Cast the label columns to float if they exist
        for col in self.label_cols:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    raise ValueError(
                        f"Column '{col}' in dataset '{dataset_name}' must be convertible to float.")
        return df

    def _normalize_log_ga_boe(self):
        """
        Normalizes the 'log_ga_boe' column in all datasets to create 'z_log_ga'.
        This method is intended to be called after loading the datasets.
        """
        # Cast the two relevant columns to float
        for col in ['ga_boe', 'log_ga_boe']:
            for name in self.dfs.keys():
                if col in self.dfs[name].columns:
                    self.dfs[name][col] = self.dfs[name][col].astype(float)

        # Standardize the log_ga_boe column in all datasets
        print(f"Using hard-coded log_ga_boe mean: {LGA_MEAN}, std: {LGA_STD}")
        for name in self.dfs.keys():
            log_ga_boe = np.log(self.dfs[name]['ga_boe'].astype(float).values)
            self.dfs[name]['z_log_ga'] = (log_ga_boe - LGA_MEAN) / LGA_STD

    def _apply_ga_filter(self):
        """
        Applies a filter to the DataFrame based on the 'ga_boe' column.
        This method filters out rows where 'ga_boe' is less than the specified
        filter_ga value. It is applied to all datasets in `self.dfs`.
        """
        if not self.filter_ga:
            return

        for name in self.dfs.keys():
            # Extract the DataFrame for the current dataset
            df = self.dfs[name]

            # Ensure the 'ga_boe' column exists
            if 'ga_boe' not in df.columns:
                raise ValueError(
                    "The 'ga_boe' column is required for filtering.")

            # Ensure the 'ga_boe' column is numeric
            if pd.api.types.is_numeric_dtype(df['ga_boe']):
                if df['ga_boe'].dtype != float:
                    try:
                        df['ga_boe'] = df['ga_boe'].astype(float)
                    except ValueError:
                        raise ValueError(
                            "The 'ga_boe' column must be of type float for filtering.")

            # Apply the filter and reset the index
            df = df[df['ga_boe'] >= self.filter_ga].reset_index(drop=True)

            # Set the reference to the filtered DataFrame
            self.dfs[name] = df

    def _apply_subset_filter(self):
        """
        Applies a filter to the DataFrame based on the binary column specified.
        """
        if not self.filter_subset:
            return

        for name in self.dfs.keys():
            # Extract the DataFrame for the current dataset
            df = self.dfs[name]

            # Ensure the specified column exists in this DataFrame
            if self.filter_subset not in df.columns:
                print(f"Warning: Column {self.filter_subset} does not " +
                      f"exist in the dataset '{name}'. Skipping filtering.")
                # Do nothing
                continue

            # Ensure the specified column can be converted to integer
            try:
                df[self.filter_subset] = df[self.filter_subset].astype(int)
            except:
                print(f"Warning: Column {self.filter_subset} is not integer" +
                      f"exist in the dataset '{name}'. Skipping filtering.")
                # Do nothing
                continue

            # Ensure the self.filter_subset column is binary
            if not np.all(np.isin(df[self.filter_subset].values, [0, 1])):
                print(f"Warning: Column {self.filter_subset} is not binary")
                # Do nothing
                continue

            # Apply the filter and reset the index
            print(f"Filtering {name} dataset on column {self.filter_subset}")
            df = df[df[self.filter_subset] == 1].reset_index(drop=True)

            # Set the reference to the filtered DataFrame
            self.dfs[name] = df

    def _create_inference_dataloader(self, dataset_name: str) -> DataLoader:
        """
        Creates a DataLoader for inference.
        """
        # Create the dataset
        dataset = VideoDatasetInference(
            df=self.dfs[dataset_name],
            path_col=self.path_col,
            label_cols=self.label_cols,
            transforms=self.transforms,
            channels=self.channels,
            data_dir=self.data_dir,
            frames_or_channel_first=self.frames_or_channel_first,
            crop_fn=self.crop_fn,
            subsample=self.subsample)

        dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=self.num_workers,
                                pin_memory=True)

        return dataloader
