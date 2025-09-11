"""
ExamDatasetFrame.py

This Dataset module is a daughter class for serving samples at an exam
level (not video level) for multiple instance learning (MIL). This daughter
class implements the __getitem__() method to load the data for MIL instances
that are frame-level.

Author: Courosh Mehanian
        Daniel Shea
        Sourabh Kulhare
        Olivia Zahn

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import random
import numpy as np
from typing import Union, Tuple

import torch

from ghlobus.utilities.sample_utils import expand_list
from ghlobus.utilities.sample_utils import matern_subsample
from ghlobus.utilities.sample_utils import random_jitter_subsample
from ghlobus.utilities.sample_utils import uniformly_distributed_subsample
from ghlobus.utilities.sweep_utils import get_known_combo_indices
from ghlobus.data.ExamDatasetBase import ExamDatasetBase


class ExamDatasetFrame(ExamDatasetBase):
    """
    Exam-based learning Dataset for TWIN Detection, e.g., using
        Multiple Instance Learning (MIL) or other methods.

    This Dataset loads frames from videos and organizes them into bags
    for Multiple Instance Learning (MIL). This class implements frame-level MIL,
    where MIL instances are frames. The class also implements various frame
    sampling strategies.

    The Dataset handles:
        - Loading and management of video frames
        - Upsampling of positive exams for class balance and GA balance
        - Multiple frame sampling strategies (random, matern, jitter, uniform)
        - Tracking of data loading errors
        - Batch organization for both frame and video-level analysis

    The output of the __getitem__() method is a tuple containing:
        - image frames (torch.Tensor)
        - label (int or float)
        - video source indices into dataframe (np.ndarray)
        - frame indices into video (np.ndarray)

    The image frames have dimensions (L, C, H, W) where:
        - L: number of frames in the bag
        - C: number of channels (1 or 3)
        - H: height of the image
        - W: width of the image

    The Lightning Dataset wrapper for this class, batches up samples
    so they have shape (B, L, C, H, W) where:
        - B: batch size

    The label is a scalar (int or float) representing the class of the bag.

    The video source indices are the indices of the videos in the original
    dataframe, and the frame indices are the indices of the frames in the
    original video. These indices have shape (L, ) where L is the bag_size.
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, float], np.ndarray, np.ndarray]:
        """
        Gets an item from the Dataset given an input index. mil_format = 'frame'

        Parameters:
            idx: int - integer specifying the index of the item in the Dataset to retrieve.

        Returns:
            item: tuple containing:
                  - image frames (torch.Tensor)
                  - label (int or float)
                  - video source indices into dataframe (np.ndarray)
                  - frame indices into video (np.ndarray)
        """
        # Get the files, indices, and tags associated with an exam index
        exam_files, exam_inds, exam_tags = self.get_exam(idx)

        # Choosing random known_combos?
        if self.random_known_combos:
            inds = get_known_combo_indices(
                exam_tags,
                allow_biometric=self.allow_biometric,
                strict_known_combos=self.strict_known_combos,
                random=True)
            exam_files = [exam_files[i] for i in inds]
            exam_inds = [exam_inds[i] for i in inds]

        # Step 1: get the frames for this exam regardless of mil_format
        exam_frames = list()
        df_inds = list()
        frame_inds = list()
        for file, file_ind in zip(exam_files, exam_inds):
            # frames = torch.load(file, map_location=self.device)
            frames = torch.load(file)

            # Check if we need to expand channel dimension
            if self.channels == 3 and frames.shape[1] == 1:
                # copy monochrome image across three channels
                frames = torch.repeat_interleave(frames, repeats=3, dim=1)
            elif self.channels == 1 and frames.shape[1] == 3:
                frames = frames[:, 0, :, :].unsqueeze(1)

            # Grab a crop near (train) or at center (val, test)
            exam_frames.append(self.crop(frames))

            # populate source indices (useful only in inference mode)
            df_inds.append(file_ind * np.ones(frames.shape[0], dtype=int))
            frame_inds.append(np.arange(frames.shape[0], dtype=int))

        # Step 2: select frames according to mil_format = 'frame'
        # Concatenate frames from all files in the exam
        exam_frames = torch.cat(exam_frames, dim=0)
        df_inds = np.concatenate(df_inds)
        frame_inds = np.concatenate(frame_inds)
        # Make sure there are exactly bag_size # of clips
        #  - If there are more clips than bag_size, choose random sample
        #  - If there are fewer clips than bag_size, repeat some
        #  - Else if there are exactly bag_size clips, Choose all (do nothing)
        if self.bag_size == exam_frames.shape[0]:
            # if exactly the right number take them all
            indices = list(range(exam_frames.shape[0]))
        elif self.bag_size > exam_frames.shape[0]:
            # if more samples requested than available, take them all then add some
            indices = expand_list(list(range(exam_frames.shape[0])),
                                  self.bag_size,
                                  max_replicates=self.max_replicates)
        # the following options are all for bag_size < exam_frames.shape[0]
        elif self.frame_sampling == 'random':
            # if number of frames exceeds bag_size, random without replacement
            indices = random.sample(range(exam_frames.shape[0]), k=self.bag_size)
        elif self.frame_sampling == 'matern':
            # use Matern sampling to get separation between samples
            indices = matern_subsample(exam_frames.shape[0], k=self.bag_size)
        elif self.frame_sampling == 'jitter':
            # uniform distribution with integer spacing and random shift
            indices = random_jitter_subsample(exam_frames.shape[0], k=self.bag_size)
        elif self.frame_sampling == 'uniform':
            # uniform distribution spread to cover entire range
            indices = uniformly_distributed_subsample(exam_frames.shape[0], k=self.bag_size)
        else:
            raise ValueError(f"Invalid frame_sampling argument {self.frame_sampling}")

        # Sort the indices to keep the frames in order
        indices.sort()
        # Grab the frames according to sampled indices
        exam_frames = exam_frames[indices, ...]
        df_inds = df_inds[indices]
        frame_inds = frame_inds[indices]

        # Step 3: Preprocess the frames
        try:
            # Apply preprocessing to the frames
            exam_frames = self.preprocess(exam_frames)
        except Exception as err:
            print("Triggered exception: ", repr(err))
            raise Exception(f"Failed self.preprocess(): {self.exams[idx]}")

        # Step 4: Augment the frames (in train mode)
        if self.mode == 'train':
            try:
                # Apply augmentations to the frames
                exam_frames = self.augment(exam_frames)
            except Exception as err:
                print("Triggered exception: ", repr(err))
                raise Exception(f"Failed self.augment(): {self.exams[idx]}")

        # Check the order of frames or channel
        if self.frames_or_channel_first == 'channel':
            exam_frames = torch.Tensor.movedim(exam_frames, 1, 0)

        # convert the frames to float and normalize
        exam_frames = exam_frames.to(dtype=torch.float32)/255.0

        # For each exam, return tuple of
        #   - image frames (torch.Tensor)
        #   - label (int or float)
        #   - video source indices into dataframe (np.ndarray)
        #   - frame indices into video (np.ndarray)
        return exam_frames, self.exam_labels[idx], df_inds, frame_inds
