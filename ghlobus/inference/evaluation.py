"""
evaluation.py

Launches the CLI for inference. This script is intended to be run from the command line.

Author: Dan Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from ghlobus.utilities.cli import InferenceCLI

InferenceCLI(save_config_kwargs={"overwrite": True, "save_to_log_dir": True})
