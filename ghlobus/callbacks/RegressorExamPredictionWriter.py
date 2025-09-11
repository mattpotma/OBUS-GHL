"""
RegressorExamPredictionWriter.py

An intermediate base class for regression-based exam prediction writers that
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

from ghlobus.callbacks.BaseExamPredictionWriter import BaseExamPredictionWriter
from ghlobus.utilities.inference_utils import find_wandb_logger
from ghlobus.utilities.plot_utils import plot_dataset_truth_vs_predictions
from ghlobus.utilities.plot_utils import plot_dataset_bland_altman
from ghlobus.utilities.plot_utils import plot_dataset_fractional_bland_altman
from ghlobus.utilities.plot_utils import plot_by_trimester


class RegressorExamPredictionWriter(BaseExamPredictionWriter):
    """
    Intermediate base class for regression-based exam prediction writers.

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
        self.error_col = f"Prediction Error ({self.unit_plot_name.lower()})"
        self.absolute_error_col = f"Absolute Error ({self.unit_plot_name.lower()})"
        self.fractional_error_col = f"Fractional Error"
        self.mae_metric_name = f"Exam MAE ({self.unit_plot_name.lower()})"

        # Common attributes for exam processing
        self.frame_features = {}
        self.mean_mae = None

    @abstractmethod
    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
                             batch: Tuple, batch_idx: int) -> None:
        """
        Abstract method to collect predictions at the end of each batch.
        Must be implemented by child classes to handle model-specific output processing.
        """
        raise NotImplementedError

    @abstractmethod
    def analyze_exam(self, model: Any, frame_features) -> Any:
        """
        Abstract method to analyze exam-level features and produce predictions.
        Must be implemented by child classes to handle model-specific analysis.
        """
        raise NotImplementedError

    def _process_and_calculate_errors(self) -> None:
        """
        Process collected predictions and calculate error metrics.

        This method assumes that predictions have been stored in exam_df
        and adds the standard error columns. err_col
        """
        # Calculate relative error
        self.exam_df[self.error_col] = (
            self.exam_df[self.predicted_col_name] -
            self.exam_df[self.label_name]
        )

        # Calculate absolute error
        self.exam_df[self.absolute_error_col] = self.exam_df[self.error_col].abs()

        # Calculate fractional error
        self.exam_df[self.fractional_error_col] = self.exam_df[self.error_col] / \
            self.exam_df[self.label_name]

        # Calculate mean absolute error
        self.mean_mae = self.exam_df[self.absolute_error_col].mean()

        print(
            f"Mean Exam-Level {self.dataset_name} {self.mae_metric_name}: {self.mean_mae}")

    def generate_summary_plots(self) -> None:
        """
        Generate standard regression plots based on configuration.
        """
        # Truth vs Predictions plot
        title = f"{self.dataset_name} Exam-Level Truth vs Prediction"
        outpath = os.path.join(self.save_dir, title.replace(' ', '_') + '.png')
        fig = plot_dataset_truth_vs_predictions(
            df=self.exam_df,
            set_name=self.dataset_name,
            label_name=self.label_name,
            label_plot_name=self.label_plot_name,
            title=title,
            unit_plot_name=self.unit_plot_name,
        )
        fig.savefig(outpath)

        # Bland-Altman plot
        title = f"{self.dataset_name} Exam-Level Bland-Altman"
        outpath = os.path.join(self.save_dir, title.replace(' ', '_') + '.png')
        if self.use_percentile_bland_altman:
            fig = plot_dataset_fractional_bland_altman(
                df=self.exam_df,
                set_name=self.dataset_name,
                label_name=self.label_name,
                err_col=self.fractional_error_col,
                label_plot_name=self.label_plot_name,
                title=title,
                unit_plot_name=self.unit_plot_name,
            )
        else:
            fig = plot_dataset_bland_altman(
                df=self.exam_df,
                set_name=self.dataset_name,
                label_name=self.label_name,
                err_col=self.error_col,
                title=title,
                by_trimester=self.include_trimester_plots,
                label_plot_name=self.label_plot_name,
                unit_plot_name=self.unit_plot_name,
            )
        fig.savefig(outpath)

        # Optional trimester plots
        if self.include_trimester_plots:
            # First, exam-level analysis plots
            title = f"{self.dataset_name} Exam-Level Analysis"
            figs = plot_by_trimester(df=self.exam_df,
                                     set_name=self.dataset_name,
                                     title=title)
            for i, fig in enumerate(figs):
                if isinstance(fig, Figure):
                    trimester_number = i + 1
                    outpath = os.path.join(
                        self.save_dir, f"{title.replace(' ', '_')}_Trimester_{trimester_number}.png")
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
            exam_table = wandb.Table(dataframe=self.exam_df)
            wandb.log({"Exam Results Table": exam_table})

            # Truth vs Predictions scatter plot
            truth_vs_pred_title = f"{self.dataset_name} Exam-Level Truth vs Prediction"
            wandb.log({
                truth_vs_pred_title: wandb.plot.scatter(
                    exam_table,
                    self.label_name,
                    self.predicted_col_name,
                    title=truth_vs_pred_title,
                    split_table=True
                )
            })

            # Bland-Altman scatter plot
            bland_altman_title = f"{self.dataset_name} Exam-Level Bland Altman"
            if self.use_percentile_bland_altman:
                wandb.log({
                    bland_altman_title: wandb.plot.scatter(
                        exam_table,
                        self.label_name,
                        self.fractional_error_col,
                        title=bland_altman_title,
                        split_table=True
                    )
                })
            else:
                wandb.log({
                    bland_altman_title: wandb.plot.scatter(
                        exam_table,
                        self.label_name,
                        self.error_col,
                        title=bland_altman_title,
                        split_table=True
                    )
                })
