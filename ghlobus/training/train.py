"""
train.py

Training script for use with PyTorch Lightning and YAML files.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import torch
from pydicom import config
from lightning.pytorch.cli import LightningCLI

#  PyDICOM validation mode spams the terminal with warnings, so disable it
config.settings.reading_validation_mode = config.IGNORE

# For the tensor cores on GPU:
torch.set_float32_matmul_precision('medium')

# Start experiment (configured via CLI arguments with Lightning Parser)
cli = LightningCLI(save_config_callback=None)
