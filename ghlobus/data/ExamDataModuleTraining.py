"""
ExamDataModuleTraining.py

MIL = Multiple Instance Learning, refers to a weakly supervised learning approach
where the labels are only available at the bag level rather than at the instance level,
each bag consisting of multiple instances, not all of which need have the same label and
whose labels are generally not known.

This Dataset module is used to dish out samples for training and validation
at an exam level (not video level). This is appropriate for models that need
exam level information during training. For example, models that use multiple
instance learning.

Four parameters are used to control the way frames are sampled:
- frame_sampling      - Temporal sampling of frames, within a video
                        or the entire exam of frames.
- use_known_combos    - Filter videos by a fixed set of known_combos,
                        suitable for inference
- random_known_combos - Randomly sample from the known_combos ON-THE-FLY,
                        suitable for training as a form of data augmentation.
- allow_biometric     - Use biometric sweeps when there aren't enough blind sweeps
                        This parameter is shared between DataModule and Dataset.
- strict_known_combos - Whether to use strict definition of known_combos
                        This parameter is shared between DataModule and Dataset.

**NOTE**  use_known_combos and random_known_combos should not both be set to True;
Either one can be set to True, or both set to False, but not both set to True.

Author: Courosh Mehanian
        Daniel Shea
        Olivia Zahn
        Sourabh Kulhare

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
# External imports
from torch.utils.data import DataLoader

# Internal imports
from ghlobus.data.ExamDataModuleBase import ExamDataModuleBase


class ExamDataModuleTraining(ExamDataModuleBase):
    """
    Exam-based learning DataModule for TWIN Detection, e.g., using
        Multiple Instance Learning (MIL) or other methods.

    This DataModule is a front-end for a Dataset object that loads frames from videos
    and organizes them into bags for Multiple Instance Learning (MIL). It supports
    frame-level, clip-level, and video-level instances, with various sampling
    strategies for frame selection.
    """
    def __init__(self, **kwargs):
        """
        Initializes the VideoDataModuleTraining.

        Parameters
        ----------
            dataset_dir: os.PathLike      - Root directory containing data split definition files
            distribution: int             - Integer indicating the distribution ID to use
            transforms: List[Callable]    - List of functions to transform the data prior
                                            to getting passed to the model
            augmentations: List[Callable] - List of functions to transform the data prior
                                            to getting passed to the model, train time only
            batch_size: int               - Size of batch for dataloaders
            num_workers: int              - Number of workers for dataloaders
            bag_size: int                 - Size of bags for dataloaders
            mil_format: str               - MIL approach: bag=exam and instance= 'video' or 'clip' or 'frame'
            channels: int                 - Number of channels for output (keep 1 or repeat to 3)
            frames: int                   - Number of sampled frames in each video (mil_format='video')
                                            Number of half-sampled frames in each clip (mil_format='clip')
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
        # Initialize the allowed_stages and corresponding datasets
        self.allowed_stages = {
            'fit': ['train', 'val'],
        }

    def train_dataloader(self) -> DataLoader:
        return self._create_train_dataloader(dataset_name='train')

    def val_dataloader(self) -> DataLoader:
        if self.use_inference_val_dataset:
            return self._create_inference_dataloader(dataset_name='val')
        return self._create_train_dataloader(dataset_name='val')
