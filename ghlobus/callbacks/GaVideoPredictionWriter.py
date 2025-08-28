"""
GaVideoPredictionWriter.py

A class that tracks video-level GA predictions and saves the results to disk.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from typing import Any, Tuple

from lightning import Trainer, LightningModule

from ghlobus.callbacks.RegressorVideoPredictionWriter import RegressorVideoPredictionWriter
from ghlobus.utilities.inference_utils import detach_and_convert_tensors
from ghlobus.utilities.inference_utils import get_rescale_log_value_func
from ghlobus.utilities.inference_utils import LGA_MEAN, LGA_STD


class GaVideoPredictionWriter(RegressorVideoPredictionWriter):
    """
    Writes predictions for the Gestational Age (GA) regression model.
    """

    def __init__(self, *args, **kwargs):
        # Hard-code GA-specific configuration
        super().__init__(
            label_name='ga_boe',
            label_plot_name='GA',
            unit_plot_name='Days',
            use_percentile_bland_altman=False,  # GA uses regular Bland-Altman plot
            include_trimester_plots=True,       # GA includes trimester analysis
            *args, **kwargs
        )

    def on_predict_batch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule,
                             outputs: Any,
                             batch: Tuple,
                             batch_idx: int,
                             dataloader_idx: int = 0) -> None:
        """
        Handles the end of a prediction batch during inference.

        Extracts the predicted value from the outputs, detaches and converts it to a
        numpy array, and stores the prediction in the `self.predictions` dictionary
        using the corresponding filename.

        Args:
            trainer (Trainer):               Lightning Trainer instance.
            pl_module (LightningModule):     Lightning model being used for prediction.
            outputs (Any):                   Outputs from the model for the current batch.
            batch (Tuple):                   Input batch data.
            batch_idx (int):                 Index of the current batch.
            dataloader_idx (int, optional):  Index of the dataloader (default is 0).

        Returns:
            None
        """
        # filename can be accessed from the dataframe using batch_idx, since batch size is 1
        filename = self.df['filename'].iloc[batch_idx]
        y_hat = outputs[0]
        [y_hat] = detach_and_convert_tensors(pl_module, [y_hat])
        self.predictions[filename] = y_hat

    def _add_custom_processing(self) -> None:
        """
        Add GA-specific processing for log GA and derived metrics.
        """
        # Add log GA predictions (this is what was stored in self.predictions)
        self.df['Predicted Z Log GA'] = self.df['filename'].map(self.predictions)

        # Create a rescale function for log GA
        print(f"Using stored log_GA mean {LGA_MEAN} and std {LGA_STD}.")
        rescale_log_ga = get_rescale_log_value_func(mean=LGA_MEAN, std=LGA_STD)

        # Calculate GA in days from log GA
        self.df[self.predicted_col_name] = self.df['Predicted Z Log GA'].apply(
            rescale_log_ga)

        # Calculate log-scale error
        self.df['Prediction Error, log'] = self.df['Predicted Z Log GA'] - \
            self.df['z_log_ga']

        # Recalculate errors using the proper GA values instead of log values
        self.df[self.error_col] = (
            self.df[self.predicted_col_name] - self.df[self.label_name]
        )
        self.df[self.absolute_error_col] = self.df[self.error_col].abs()
        self.df[self.fractional_error_col] = self.df[self.error_col] / self.df[self.label_name]

        # Recalculate mean MAE with proper values
        self.mean_mae = self.df[self.absolute_error_col].mean()
