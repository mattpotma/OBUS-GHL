"""
expt_config.py

Useful function definitions.

Author: Daniel Shea.

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from torchvision.transforms import v2 as transforms

# Image final dimensions and dimensions after blank padding
IMG_DIMS = (256, 256)

# DataModule helper functions:
centercrop = transforms.CenterCrop(IMG_DIMS)
randomcrop = transforms.RandomCrop(IMG_DIMS,
                                   padding=0,
                                   pad_if_needed=True)
