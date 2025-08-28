"""
VideoDataModuleInference.py

This module defines the VideoDataModuleInference class, which creates a DataLoader for
use in the inference process. It is designed to be flexible, supporting both
GA and FP inference tasks by allowing for different dataset classes and
configurations.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from typing import Any, Literal, Callable

from pandas import DataFrame
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import RandomCrop

from ghlobus.data.VideoDataModuleBase import VideoDataModuleBase


class VideoDataModuleInference(VideoDataModuleBase):
    """
    A DataModule for video inference, supporting GA, FP, and EFW tasks.
    """

    def __init__(self,
                 dataset_name: Literal["train", "val", "test"],
                 **kwargs: Any,
                 ) -> None:
        """
        Initializes the VideoDataModuleInference.

        Parameters
        ----------
            dataset_name: Literal              Name of the dataset to use ("train", "val", or "test").
            dataset_cls: Type[Dataset]         Dataset class to use for creating the dataset.
            csv_path_template: os.PathLike     Template for the CSV path.
            transforms: List[Callable]         List of transforms to apply to the data.
            label_cols: List[str], optional    List of columns to use as labels.
            filter_ga: float, optional         GA cut-off value for filtering the data.
        """
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.allowed_stages = {
            'predict': [dataset_name],
        }
        self._df = None
        self._crop_fn = RandomCrop(
            self.image_dims) if self.image_dims else None

    @property
    def df(self) -> DataFrame:
        """
        Returns the DataFrame for the specified dataset name.
        """
        if self._df is None:
            self._df = self.dfs.get(self.dataset_name, None)
        return self._df

    @property
    def crop_fn(self) -> Callable:
        """
        Returns the crop function for the dataset.
        """
        return self._crop_fn

    def predict_dataloader(self) -> DataLoader:
        return self._create_inference_dataloader(self.dataset_name)
