"""
ExamDatasetBase.py

This Dataset module is the abstract base class for serving samples at an exam
level (not video level). This is appropriate for training and inferencing models
that need exam level information during training and inference. For example,
models that use multiple instance learning, where the bag is the exam. Various
daughter classes implement the __getitem__() method to load the data for MIL
instances that are frame-level, clip-level, or video-level.

Author: Courosh Mehanian
        Sourabh Kulhare
        Daniel Shea
        Olivia Zahn

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import random
import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm
from operator import itemgetter
from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import RandomCrop
from torchvision.transforms.v2 import CenterCrop

from ghlobus.utilities.sample_utils import expand_list
from ghlobus.utilities.constants import GA_BINS, GA_LOW, GA_HIGH, GA_WIDTH


# noinspection PyUnresolvedReferences
class ExamDatasetBase(Dataset, ABC):
    """
    Exam-based learning Dataset for TWIN Detection, e.g., using
        Multiple Instance Learning (MIL) or other methods.

    This Dataset loads frames from videos and organizes them into bags
    for Multiple Instance Learning (MIL). It supports frame-level, clip-level, and
    video-level instances, with various sampling strategies for frame selection.
    This is an abstract class, and children of this class must implement the
    __getitem__() method.

    The Dataset handles:
        - Loading and management of video frames
        - Upsampling of positive exams for class balance and GA balance
        - Multiple frame sampling strategies (random, matern, jitter, uniform)
        - Tracking of data loading errors
        - Batch organization for both frame and video-level analysis

    Parameters:
        df: pd.DataFrame              - DataFrame with the following columns:
            exam_dir: str             - Name of the exam folder (defines "exam")
            outpath: str              - Absolute paths of dataset videos
            TWIN: int                 - Video labels in {0, 1}
            GA: float                 - Gestational Age of fetus at exam date (days)
            tag: str                  - Video sweep tags (e.g., M, L, R, BPD, etc.)
        mode: str                     - train, val, or test (batch_size=1)
        transforms: List[Callable]    - a list of functions to transform the data prior
                                        to getting passed to the model
        augmentations: List[Callable] - a list of functions to augment the data
        bag_size: int                 - the size of the bag
        mil_format: str               - Either 'frame', 'clip', or 'video'
        channels: int                 - number of channels for output (keep 1 or repeat to 3)
        frames: int                   - number of sample frames in each video (mil_format='video')
        frame_sampling: str           - Either 'random', 'jitter', 'uniform', or 'matern'
        frames_or_channel_first: str  - either 'channel' or 'frame'
        random_known_combos: bool     - Whether to randomly sample from KNOWN_COMBOS (**on-the-fly**)
        allow_biometric: bool         - Whether to use biometric sweeps when insufficient KNOWN_COMBOS
        strict_known_combos: bool     - Whether to use strict definition of KNOWN_COMBOS
        upsample: bool                - whether to upsample the exams to balance categories
        max_replicates: int           - maximum number of times to replicate an exam
        balance_ga: bool              - Whether to balance GA when upsampling exams
        image_dims: int or (int, int) - size of output image for inference. CenterCrop
                                       is performed on the data to the given image dims.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 mode: str,
                 transforms: List[Callable],
                 augmentations: List[Callable],
                 bag_size: int,
                 mil_format: str = 'frame',
                 channels: int = 3,
                 frames: int = 50,
                 frame_sampling: str = 'matern',
                 frames_or_channel_first: str = 'frames',
                 random_known_combos: bool = False,
                 allow_biometric: bool = False,
                 strict_known_combos: bool = False,
                 upsample: bool = True,
                 max_replicates: int = 40,
                 balance_ga: bool = False,
                 image_dims: Union[int, Tuple[int, int]] = (256, 256)
                 ) -> None:
        # Call the parent constructor
        super(ExamDatasetBase, self).__init__()

        # Initialize attributes
        self.df = df

        # Double up image_dims, if needed
        if isinstance(image_dims, int):
            self.image_dims = (image_dims, image_dims)
        else:
            self.image_dims = image_dims
        # Grab a crop at center (val, test), or near center (train)
        if mode in ['val', 'test']:
            self.crop = CenterCrop(self.image_dims)
        elif mode == 'train':
            self.crop = RandomCrop(self.image_dims,
                                   padding=0,
                                   pad_if_needed=True)
        else:
            raise ValueError(f"mode {mode} is not supported")
        self.mode = mode
        self.bag_size = bag_size
        self.mil_format = mil_format
        self.transforms = list(transforms)
        self.augmentations = list(augmentations)
        self.channels = channels
        self.frames = frames
        self.frame_sampling = frame_sampling
        self.frames_or_channel_first = frames_or_channel_first
        self.random_known_combos = random_known_combos
        self.allow_biometric = allow_biometric
        self.strict_known_combos = strict_known_combos
        self.max_replicates = max_replicates
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Relating to exam-level training or inference
        self.video_labels = self.df['TWIN'].astype(int).tolist()
        self.video_gas = self.df['GA'].astype(float).tolist()
        self.exams = list(set(self.df['exam_dir'].tolist()))
        self.exam_labels = list()
        self.exam_gas = list()
        self.initialize_metadata()

        # Plain exam upsampling,upsampling while balancing GA, or none
        if upsample and balance_ga:
            self.upsample_exams_by_ga()
        elif upsample:
            self.upsample_exams()
        self.shuffle_exams()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, float], np.ndarray, np.ndarray]:
        """
        Gets an item from the Dataset given an input index. This method is abstract;
        Child classes must implement this method.

        Parameters:
            idx: int - integer specifying the index of the item in the Dataset to retrieve.

        Returns:
            item: tuple containing:
                  - image frames (torch.Tensor)
                  - label (int or float)
                  - video source indices into dataframe (np.ndarray)
                  - frame indices into video (np.ndarray)
        """

    def __len__(self) -> int:
        return len(self.exams)

    def get_exam(self, idx: int) -> Tuple[List[str], List[int], List[str]]:
        """
        Gets an exam from the Dataset given an input index.

        Parameters:
            idx: int       - integer specifying index exam to retrieve.

        Returns:
            items: tuple   - consisting of
                             exam_files (List[str])
                             exam_inds (List[int])
                             and exam_tags (List[str])
        """
        # Get the exam_dir aka exam_id
        exam = self.exams[idx]
        # Get the exam and find files, indices, and tags associated with it
        this_exam = self.df['exam_dir'] == exam
        exam_files = self.df[this_exam]['outpath'].tolist()
        exam_tags = self.df[this_exam]['tag'].tolist()
        exam_inds = self.df.index[this_exam].tolist()

        return exam_files, exam_inds, exam_tags

    def upsample_exams(self) -> None:
        """
        Balance the dataset by upsampling positive examples.

        Duplicates positive exam samples to match the number of negative examples,
        preventing class imbalance during training. Updates self.exams and
        self.exam_labels to include the upsampled data.
        """
        # Find exams with label = 1
        positive_exams = [exam for exam, label in zip(self.exams, self.exam_labels) if label == 1]
        negative_exams = [exam for exam, label in zip(self.exams, self.exam_labels) if label == 0]
        print(f"Number of positive exams: {len(positive_exams)}")
        print(f"Number of negative exams: {len(negative_exams)}")
        print(f"Number of exams: {len(negative_exams) + len(positive_exams)}")

        # Calculate the number of positive exams needed to match the number of negative exams
        print("Upsampling exams...")
        upsampled_positive_exams = expand_list(positive_exams,
                                               len(negative_exams),
                                               max_replicates=self.max_replicates)

        # Update exams and exam_labels
        self.exams = negative_exams + upsampled_positive_exams
        self.exam_labels = [0] * len(negative_exams) + [1] * len(upsampled_positive_exams)

        print(f"Number of positive exams after upsampling: {sum(self.exam_labels)}")
        print(f"Number of negative exams after upsampling: {len(self.exams) - sum(self.exam_labels)}")
        print(f"Number of exams after upsampling: {len(self.exams)}")

    def upsample_exams_by_ga(self) -> None:
        """
        Balance the dataset by upsampling positive examples, also accounting for GA balance.

        Makes a histogram of positive and negative exams by binning GA into 4 bins.
        Duplicates both negative and positive exam samples to match the largest bin count,
        preventing class and GA imbalance during training. Updates self.exams and
        self.exam_labels to include the upsampled data.
        """
        # digitize GA
        bins = np.arange(GA_LOW, GA_HIGH+1, GA_WIDTH)
        adj_gas = np.minimum(np.maximum(GA_LOW, self.exam_gas), GA_HIGH)
        exam_dig_gas = np.minimum(np.digitize(adj_gas, bins, False)-1, GA_BINS-1)

        # Convert lists to numpy arrays for easy masking
        exams = np.array(self.exams)
        labels = np.array(self.exam_labels)
        gas = np.array(exam_dig_gas)

        # Get positive and negative masks
        pos_mask = np.array(labels == 1)
        neg_mask = np.array(labels == 0)

        # Construct digitized GA counter for positive and negative exams
        pos_exams = exams[pos_mask]
        neg_exams = exams[neg_mask]
        pos_gas = gas[pos_mask]
        neg_gas = gas[neg_mask]
        # Get the bin counts for positive and negative exams
        pos_bincounts = np.bincount(pos_gas)
        neg_bincounts = np.bincount(neg_gas)
        print(f"Number of positive exams: {len(pos_gas)}")
        print(f"Positive bin counts: {pos_bincounts}")
        print(f"Number of negative exams: {len(neg_gas)}")
        print(f"Negative bin counts: {neg_bincounts}")
        print(f"Number of exams: {len(pos_gas) + len(neg_gas)}")

        # Get the largest histogram and duplicate to this one
        max_bincount = max(np.max(pos_bincounts), np.max(neg_bincounts))

        # Upsample the positive and negative exams
        print("Upsampling exams...")
        upsampled_pos_exams = list()
        upsampled_pos_counts = list()
        upsampled_neg_exams = list()
        upsampled_neg_counts = list()
        for z in range(GA_BINS):
            z_pos_count = pos_bincounts[z]
            z_pos_mask = np.array(pos_gas == z)
            z_pos_exams = list(pos_exams[z_pos_mask])
            if z_pos_count < max_bincount:
                z_pos_exams = expand_list(z_pos_exams,
                                          max_bincount,
                                          max_replicates=self.max_replicates)
            upsampled_pos_counts.append(len(z_pos_exams))
            upsampled_pos_exams.extend(z_pos_exams)
            z_neg_count = neg_bincounts[z]
            z_neg_mask = np.array(neg_gas == z)
            z_neg_exams = list(neg_exams[z_neg_mask])
            if z_neg_count < max_bincount:
                z_neg_exams = expand_list(z_neg_exams,
                                          max_bincount,
                                          max_replicates=self.max_replicates)
            upsampled_neg_counts.append(len(z_neg_exams))
            upsampled_neg_exams.extend(z_neg_exams)

        # Report new statistics
        print(f"Number of positive exams after upsampling: {len(upsampled_pos_exams)}")
        print(f"Positive bin counts after upsampling: {upsampled_pos_counts}")
        print(f"Number of negative exams after upsampling: {len(upsampled_neg_exams)}")
        print(f"Negative bin counts after upsampling: {upsampled_neg_counts}")

        # Update exams and exam_labels
        self.exams = upsampled_neg_exams + upsampled_pos_exams
        self.exam_labels = [0] * len(upsampled_neg_exams) + [1] * len(upsampled_pos_exams)
        print(f"Number of exams after upsampling: {len(self.exams)}")

        # Reset exam_gas to None
        self.exam_gas = None

    def shuffle_exams(self) -> None:
        """
        Randomly shuffle the exam list and their corresponding labels.

        Maintains the pairing between exams and their labels while
        randomizing their order in the dataset.
        """
        order = list(range(len(self.exams)))
        random.shuffle(order)
        self.exams = list(itemgetter(*order)(self.exams))
        self.exam_labels = list(itemgetter(*order)(self.exam_labels))

    def initialize_metadata(self) -> None:
        """
        Computes exam_labels and exam_GAs by inspecting the TWIN and GA columns
        of the dataframe and grouping by exams.
        """
        print("Generating exam labels and GAs ...")

        # Group by exam_dir and store a single label if all labels match, or None otherwise
        grouped_labels = (
            self.df
            .groupby('exam_dir')['TWIN']
            .apply(lambda x: x.iloc[0] if len(set(x)) == 1 else None)
        )

        # Group by exam_dir and store GA
        grouped_gas = (
            self.df
            .groupby('exam_dir')['GA']
            .apply(lambda x: x.iloc[0] if len(set(x)) == 1 else None)
        )

        # Accumulate labels from the grouped_labels
        for exam in tqdm(self.exams):
            # Get the exam label
            label = int(grouped_labels[exam])
            # Check if there were inconsistent video labels in the exam
            if label is None:
                # Grab all video labels for reporting
                video_labels = self.video_labels[self.df['exam_dir'] == exam].tolist()
                print(f"Labels for exam {exam} are not the same: {video_labels}")
                # Grab the majority label
                label = scipy.stats.mode(video_labels, keepdims=False).mode
            # store the exam label
            self.exam_labels.append(label)
            # Get the exam GA
            ga = float(grouped_gas[exam])
            if ga is None:
                # Grab all video GAs for reporting
                video_gas = self.video_gas[self.df['exam_dir'] == exam].tolist()
                print(f"GAs for exam {exam} are not the same: {video_gas}")
                # Grab the mean GA
                ga = np.mean(video_gas)
            # store the exam GA
            self.exam_gas.append(ga)

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply the transforms in self.transforms to the input frames.

        Parameters:
            frames: torch.Tensor - data Tensor containing frame data.

        Returns:
            frames: torch.Tensor - data modified by the self.transforms functions.
        """
        # Must collapse bag dimension in mil_format video or clip mode
        if self.mil_format == 'video' or self.mil_format == 'clip':
            K, L, C, H, W = frames.size()
            frames = frames.view(K * L, C, H, W)
        for transform in self.transforms:
            frames = transform(frames)
        # Must un-collapse bag dimension in mil_format video or clip mode
        if self.mil_format == 'video' or self.mil_format == 'clip':
            # noinspection PyUnboundLocalVariable
            frames = frames.view(K, L, C, H, W)

        return frames

    def augment(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply the augmentations in self.augmentations to the input frames.

        Parameters:
            frames: torch.Tensor - data Tensor containing frame data.

        Returns:
            frames: torch.Tensor - data modified by the self.augmentations functions.
        """
        # Must collapse bag dimension in mil_format video or clip mode
        if self.mil_format == 'video' or self.mil_format == 'clip':
            K, L, C, H, W = frames.size()
            frames = frames.view(K * L, C, H, W)
        # randomly choose an augmentation to apply
        if self.augmentations:
            random_aug = random.choice(self.augmentations)
            frames = random_aug(frames)
        # Must un-collapse bag dimension in mil_format video or clip mode
        if self.mil_format == 'video' or self.mil_format == 'clip':
            # noinspection PyUnboundLocalVariable
            frames = frames.view(K, L, C, H, W)

        return frames
