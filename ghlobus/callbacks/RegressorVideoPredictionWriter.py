"""
RegressorVideoPredictionWriter.py

An intermediate base class for regression-based video prediction writers that
extracts shared logic for units, plotting parameters, and common processing patterns.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import wandb
from abc import abstractmethod
from typing import Any, Tuple

from matplotlib.figure import Figure
from lightning import Trainer, LightningModule

from ghlobus.callbacks.BaseVideoPredictionWriter import BaseVideoPredictionWriter
from ghlobus.utilities.inference_utils import find_wandb_logger
from ghlobus.utilities.plot_utils import plot_dataset_truth_vs_predictions
from ghlobus.utilities.plot_utils import plot_dataset_bland_altman
from ghlobus.utilities.plot_utils import plot_dataset_fractional_bland_altman
from ghlobus.utilities.plot_utils import plot_by_trimester


class RegressorVideoPredictionWriter(BaseVideoPredictionWriter):
    """
    Intermediate base class for regression-based video prediction writers.

    This class extracts common patterns from EFW and GA prediction writers,
    including unit handling, plot generation, and error calculation logic.
    """

    def __init__(self,
                 label_name: str,
                 label_plot_name: str,
                 unit_plot_name: str,
                 use_percentile_bland_altman: bool = False,
                 include_trimester_plots: bool = False,
                 *args, **kwargs):
        """
        Initialize the regressor prediction writer.

        Parameters
        ----------
        label_name : str                    Column name for ground truth values (e.g., 'EFW', 'ga_boe')
        label_plot_name : str               Display name for plots (e.g., 'EFW', 'GA')
        unit_plot_name : str                Unit name for plots (e.g., 'g', 'Days')
        use_percentile_bland_altman : bool  Whether to use percentile Bland-Altman plot. Default: False
        include_trimester_plots : bool      Whether to generate trimester plots. Default: False
        """
        super().__init__(*args, **kwargs)

        # Store configuration as attributes
        self.label_name = label_name
        self.label_plot_name = label_plot_name
        self.unit_plot_name = unit_plot_name
        self.use_percentile_bland_altman = use_percentile_bland_altman
        self.include_trimester_plots = include_trimester_plots

        # Derive column names from label and unit names
        self.predicted_col_name = f"Predicted {self.label_plot_name} ({self.unit_plot_name})"
        self.error_col = f"Prediction Error ({self.unit_plot_name})"
        self.fractional_error_col = f"Fractional Error"
        self.absolute_error_col = f"Absolute Error ({self.unit_plot_name})"
        self.mae_metric_name = f"MAE ({self.unit_plot_name})"

        self.mean_mae = None

    @abstractmethod
    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
                             batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Abstract method to collect predictions at the end of each batch.
        Must be implemented by child classes to handle model-specific output processing.
        """
        raise NotImplementedError

    def _process_and_save_predictions(self) -> None:
        """
        Process collected predictions and calculate error metrics.

        This method assumes that predictions have been stored in self.predictions
        and adds the standard error columns to the dataframe.
        """
        # Map predictions to the dataframe
        self.df[self.predicted_col_name] = self.df['filename'].map(
            self.predictions)

        # Calculate relative error
        self.df[self.error_col] = (
            self.df[self.predicted_col_name] - self.df[self.label_name]
        )

        # Calculate absolute error
        self.df[self.absolute_error_col] = self.df[self.error_col].abs()

        # Calculate fractional error
        self.df[self.fractional_error_col] = self.df[self.error_col] / self.df[self.label_name]

        # Calculate mean absolute error
        self.mean_mae = self.df[self.absolute_error_col].mean()

        print(f"Mean Video-level {self.dataset_name} " +
              f"{self.mae_metric_name}: {self.mean_mae}")

        # Hook for custom processing
        self._add_custom_processing()

    def generate_summary_plots(self) -> None:
        """
        Generate standard regression plots based on configuration.
        """
        # Truth vs Predictions plot
        title = f"{self.dataset_name} Video-Level Truth vs Prediction"
        outpath = os.path.join(self.save_dir, title.replace(' ', '_') + '.png')
        fig = plot_dataset_truth_vs_predictions(
            df=self.df,
            set_name=self.dataset_name,
            label_name=self.label_name,
            label_plot_name=self.label_plot_name,
            title=title,
            unit_plot_name=self.unit_plot_name,
        )
        fig.savefig(outpath)

        # Bland-Altman plot
        title = f"{self.dataset_name} Video-Level Bland-Altman"
        outpath = os.path.join(self.save_dir, title.replace(' ', '_') + '.png')
        if self.use_percentile_bland_altman:
            fig = plot_dataset_fractional_bland_altman(
                df=self.df,
                set_name=self.dataset_name,
                label_name=self.label_name,
                err_col=self.fractional_error_col,
                label_plot_name=self.label_plot_name,
                title=title,
                unit_plot_name=self.unit_plot_name,
            )
        else:
            fig = plot_dataset_bland_altman(
                df=self.df,
                set_name=self.dataset_name,
                label_name=self.label_name,
                err_col=self.error_col,
                title=title,
                label_plot_name=self.label_plot_name,
                unit_plot_name=self.unit_plot_name,
            )
        fig.savefig(outpath)

        # Optional trimester plots
        if self.include_trimester_plots:
            title = f"{self.dataset_name} Video-Level Analysis"
            figs = plot_by_trimester(df=self.df,
                                     set_name=self.dataset_name,
                                     title=title)
            for i, fig in enumerate(figs):
                if isinstance(fig, Figure):
                    outpath = os.path.join(self.save_dir,
                                           f"{title.replace(' ', '_')}_Trimester_{i+1}.png")
                    fig.savefig(outpath)

    def log_to_wandb(self, trainer: Trainer) -> None:
        """
        Log results to Weights & Biases using configuration parameters.
        """
        wb_logger = find_wandb_logger(trainer)
        if wb_logger:
            # Log basic metrics
            wb_logger.log_hyperparams({"Dataset": self.dataset_name})
            wb_logger.log_hyperparams({self.mae_metric_name: self.mean_mae})

            # Create and log data table
            wandb_table = wandb.Table(dataframe=self.df)
            wandb.log({"Video Results Table": wandb_table})

            # Truth vs Predictions scatter plot
            truth_vs_pred_title = f"{self.dataset_name} Video-Level Truth vs Prediction"
            wandb.log({
                truth_vs_pred_title: wandb.plot.scatter(
                    wandb_table,
                    self.label_name,
                    self.predicted_col_name,
                    title=truth_vs_pred_title
                )
            })

            # Bland-Altman scatter plot
            bland_altman_title = f"{self.dataset_name} Video-Level Bland-Altman"
            if self.use_percentile_bland_altman:
                # Fractional error Bland-Altman scatter plot
                wandb.log({
                    bland_altman_title: wandb.plot.scatter(
                        wandb_table,
                        self.label_name,
                        self.fractional_error_col,
                        title=bland_altman_title
                    )
                })
            else:
                # Bland-Altman scatter plot
                wandb.log({
                    bland_altman_title: wandb.plot.scatter(
                        wandb_table,
                        self.label_name,
                        self.error_col,
                        title=f"{bland_altman_title} ({self.unit_plot_name})",
                    )
                })


    def _add_custom_processing(self) -> None:
        """
        Hook for subclasses to add custom processing after standard processing.
        Override this method if additional processing is needed.
        """
        pass
