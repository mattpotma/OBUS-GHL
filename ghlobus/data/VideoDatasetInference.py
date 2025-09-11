"""
VideoDatasetBase.py

Video Datasets for Training and Inference of GA, FP, and EFW models.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from typing import Tuple
from torch import Tensor

from ghlobus.data.VideoDatasetBase import VideoDatasetBase


class VideoDatasetInference(VideoDatasetBase):
    """
    A flexible inference dataset that can be configured to handle different
    DataFrame structures and subsampling strategies.

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

    def _process_frames(self, frames: Tensor) -> Tensor:
        """
        Applies inference-specific processing to frames.
        """
        return frames

    def _get_batch(self, frames: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns the final batch tuple.
        """
        return frames, labels
