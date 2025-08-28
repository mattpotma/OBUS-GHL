"""
ClassificationVideoPredictionWriter.py

A class that tracks video-level Fetal Presentation predictions and saves the results to disk.

Author: Daniel Shea
        Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import wandb
import numpy as np
import pandas as pd
from typing import Any, Tuple
from sklearn.metrics import classification_report
from lightning import Trainer, LightningModule
from ghlobus.callbacks.BaseVideoPredictionWriter import BaseVideoPredictionWriter
from ghlobus.utilities.inference_utils import detach_and_convert_tensors
from ghlobus.utilities.inference_utils import find_wandb_logger
from ghlobus.utilities.plot_utils import plot_auroc_curve
from ghlobus.utilities.constants import SOFTMAX_NEG, SOFTMAX_POS


class ClassificationVideoPredictionWriter(BaseVideoPredictionWriter):
    """
    Callback class for classification video prediction writing.

    This class extracts common patterns from FP and TWIN (BAA) prediction writers,
    including unit handling, plot generation, and error calculation logic.

    Attributes
    ----------
        logits : dict            Dictionary to store logits for each video.
        softmax_prob_neg : dict  Dictionary to store softmax probabilities for the negative class.
        softmax_prob_pos : dict  Dictionary to store softmax probabilities for the positive class.
        """

    def __init__(self, *args, **kwargs):
        """
        Constructs the ClassificationVideoPredictionWriter.

        Parameters
        ----------
            save_dir : os.PathLike   Directory to save output.
            save_plots : bool        If True, generate and save summary plots.
            class_names : list[str]  List of class names for classification tasks.
            label_col : str          Column name for labels in the dataframe.
            write_interval : str     When to write predictions ('batch' or 'epoch').
        """
        super().__init__(*args, **kwargs)
        self.logits = {}
        self.softmax_prob_neg = {}
        self.softmax_prob_pos = {}

    def on_predict_batch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule,
                             outputs: Any,
                             batch: Tuple,
                             batch_idx: int,
                             dataloader_idx: int = 0,
                             ) -> None:
        """
        This method logs the predictions for each video in the batch.

        Parameters
        ----------
            trainer : Trainer            Trainer instance.
            pl_module : LightningModule  Lightning module instance.
            outputs : Any                outputs from the model.
            batch : Tuple                Input batch.
            batch_idx : int              Index of the batch.
            dataloader_idx : int         Index of the dataloader (default is 0).
        """
        # filepath is from the dataframe using batch_idx, since batch size is 1
        filename = self.df['filename'].iloc[batch_idx]
        y_hat, _, _, _, logits = outputs
        [y_hat, logits] = detach_and_convert_tensors(
            pl_module, [y_hat, logits])

        # Store predictions, logits, and probabilities
        self.predictions[filename] = np.argmax(np.exp(y_hat)).item()
        self.logits[filename] = logits
        self.softmax_prob_neg[filename] = np.exp(y_hat[0][0])
        self.softmax_prob_pos[filename] = np.exp(y_hat[0][1])

    def _process_and_save_predictions(self) -> None:
        """
        Process collected predictions and calculate error metrics.

        This method assumes that predictions have been stored in self.predictions
        and adds the standard error columns to the dataframe.
        """
        # Ensure the DataFrame has the necessary columns
        self.df['Predicted label'] = self.df['filename'].map(self.predictions)
        self.df['Logits'] = self.df['filename'].map(self.logits)
        self.df[SOFTMAX_NEG] = self.df['filename'].map(
            self.softmax_prob_neg)
        self.df[SOFTMAX_POS] = self.df['filename'].map(
            self.softmax_prob_pos)

        # Build classification report and save to CSV
        report = classification_report(
            self.df[self.label_col],
            self.df['Predicted label'],
            target_names=self.class_names,
            output_dict=True,
        )
        report_df = pd.DataFrame(report).transpose()
        outpath = os.path.join(
            self.save_dir, f"{self.dataset_name}_video_classification_report.csv")
        report_df.to_csv(outpath, index=False)
        print(f"Classification report saved to: {outpath}")

    def generate_summary_plots(self) -> None:
        """
        This method creates an ROC plot to be saved to local disk.
        """
        # Generate the ROC curve plot for video-level predictions and save it
        title = f"{self.dataset_name} Video-Level ROC Curve"
        outpath = os.path.join(self.save_dir, title.replace(' ', '_') + '.png')
        fig = plot_auroc_curve(
            df=self.df,
            title=title,
            label_col=self.label_col,
        )
        fig.savefig(outpath)

    def log_to_wandb(self, trainer: Trainer) -> None:
        """
        This method logs the exam results to Weights & Biases (W&B).

        Parameters
        ----------
            trainer : Trainer  Trainer instance.
        """
        # Log the video-level results to W&B
        wb_logger = find_wandb_logger(trainer)
        if wb_logger:
            wb_logger.log_hyperparams({"Dataset": self.dataset_name})
            wandb_table = wandb.Table(dataframe=self.df)
            wandb.log({"Video Results Table": wandb_table})

            softmax_output = np.stack([
                self.df[SOFTMAX_NEG].to_numpy(),
                self.df[SOFTMAX_POS].to_numpy(),
            ], axis=1)

            wandb.log({
                f"{self.dataset_name} Video-level ROC curve": wandb.plot.roc_curve(
                    self.df[self.label_col].tolist(),
                    softmax_output,
                    labels=self.class_names,
                    title=f"{self.dataset_name} Video-level ROC curve"
                )
            })
