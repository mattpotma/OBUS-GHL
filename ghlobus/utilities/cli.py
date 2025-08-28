"""
evaluation.py

Defines the CLI class for inference. This class is instantiated
in the evaluation script.

Author: Dan Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from typing import Dict, Any
from lightning.pytorch.cli import LightningCLI
from ghlobus.utilities.inference_utils import find_wandb_logger
from ghlobus.utilities.inference_utils import yaml_to_dict


DEFAULT_EXPERIMENT_NAME = "test_output"
DEFAULT_OUTPUT_DIR = f"./{DEFAULT_EXPERIMENT_NAME}"


class InferenceCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        """
        Adds custom arguments to the parser.

        Args:
            parser: The argument parser to which arguments are added.

        Returns:
            None
        """
        parser.add_argument("--name", default=DEFAULT_EXPERIMENT_NAME)
        parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

        # link output_dir to the other variables
        # For the vector and prediction writers
        parser.link_arguments(
            "output_dir", "trainer.callbacks.init_args.save_dir")
        # For W&B
        parser.link_arguments("name", "trainer.logger.init_args.name")
        parser.link_arguments(
            "output_dir", "trainer.logger.init_args.save_dir")
        # For lightning outputs
        parser.link_arguments("output_dir", "trainer.default_root_dir")

    def before_predict(self) -> None:
        """
        Executes code before the `predict` action.

        This method ensures that the Cnn2RnnRegressor returns intermediates and reports config to W&B, if there is a wb logger.

        Returns:
            None
        """
        # Code before `predict` action

        # 1. Ensure the Cnn2RnnRegressor returns intermediates
        # (required for saving the feature, attention, and context vectors)
        # i.e. model.forward -> (y_hat, feature vectors, attention, context)
        self.model.report_intermediates = True

        # Report config to W&B, if there is a wb logger
        wblogger = find_wandb_logger(self.trainer)
        if wblogger:
            # Reports list of config yaml file
            config_files = self._get(self.config, 'config')
            # Assume we care most about the first config file,
            # and get the absolute path of the config file
            yaml_file_path = config_files[0].absolute
            # Load JSON data from file
            data = yaml_to_dict(yaml_file_path)
            # update the config with the yaml data:
            wblogger.log_hyperparams(data)
    
    def _prepare_subcommand_kwargs(self, subcommand: str) -> Dict[str, Any]:
        """
        Prepares keyword arguments for the subcommand.

        Args:
            subcommand: The subcommand to prepare arguments for.

        Returns:
            A dictionary of keyword arguments.
        """
        # Prepare the keyword arguments for the subcommand from the parent class
        kwargs = super()._prepare_subcommand_kwargs(subcommand)
        # Note - need to ensure that `predict` subcommand has the 
        # kwarg `return_predictions=False`, 
        # so accumulated output predictions are NOT accumulated, and NOT returned. 
        # This prevents memory issues with large datasets.        
        if subcommand == "predict":
            kwargs["return_predictions"] = False
        return kwargs