"""
ClassificationExamPredictionWriter.py

A class used to perform exam-level fetal presentation analysis.

Author: Daniel Shea
        Olivia Zahn
        Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""

import os
from typing import Any, Tuple, List, Union, Dict

import wandb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from lightning import Trainer, LightningModule

from ghlobus.callbacks.BaseExamPredictionWriter import BaseExamPredictionWriter
from ghlobus.utilities.inference_utils import detach_and_convert_tensors
from ghlobus.utilities.inference_utils import find_wandb_logger
from ghlobus.utilities.sweep_utils import DEFAULT_EXAM_COLS
from ghlobus.utilities.sweep_utils import KNOWN_COMBO_TAGS_FP
from ghlobus.utilities.sweep_utils import COLUMNS_TO_RECORD
from ghlobus.utilities.plot_utils import plot_auroc_curve
from ghlobus.utilities.constants import SOFTMAX_NEG, SOFTMAX_POS


class ClassificationExamPredictionWriter(BaseExamPredictionWriter):
    """
    A class used to perform exam-level classification model analysis.

    Attributes
    ----------
        logits : dict                    Dictionary to store logits for each video.
        known_tag_combos : str or None   Path to the known tag combinations file, if applicable.
    """

    def __init__(self,
                 save_dir: os.PathLike,
                 max_instances_per_exam: Union[int, None] = None,
                 use_known_tag_combos: bool = False,
                 default_exam_cols: List[str] = DEFAULT_EXAM_COLS,
                 columns_to_record: Union[List[str], Dict] = COLUMNS_TO_RECORD,
                 save_plots: bool = False,
                 class_names: Union[List[str], None] = None,
                 label_col: Union[str, None] = None,
                 feature_name: Union[str, None] = None,
                 write_interval: str = "epoch") -> None:
        """
        Constructs all the necessary attributes for the BaseExamPredictionWriter object.

        Parameters
        ----------
            save_dir : os.PathLike                Directory where the output will be saved.
            max_instances_per_exam : None or int  Upper limit of the number of videos per exam to
                                                  consider. Set to None to consider all videos.
            use_known_tag_combos : bool           Whether to use the known_tag_combos list.
            default_exam_cols : List[str]         Default columns defining an exam.
            columns_to_record : List[str]         Columns to record in the exam-level predictions.
            save_plots : bool                     Boolean indicating whether summary plots should be generated.
            class_names : list[str]               List of class names for classification tasks.
            label_col : str                       Column name for labels in the dataframe.
            feature_name : str                    Name of the feature to be used in the model.
            write_interval : str                  Interval at which the output will be written.
        """
        # Determine the columns to record based on the feature name
        super().__init__(save_dir,
                         max_instances_per_exam,
                         use_known_tag_combos,
                         default_exam_cols,
                         columns_to_record[feature_name],
                         save_plots,
                         class_names,
                         label_col,
                         write_interval)
        self.logits = {}
        self.known_tag_combos = KNOWN_COMBO_TAGS_FP if use_known_tag_combos else None

    def on_predict_batch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule,
                             outputs: Any,
                             batch: Tuple,
                             batch_idx: int) -> None:
        """
        This method logs the predictions for each video in the batch.

        Parameters
        ----------
            trainer : Trainer            Trainer instance.
            pl_module : LightningModule  Lightning module instance.
            outputs : Any                outputs from the model.
            batch : Tuple                Input batch.
            batch_idx : int              Index of the batch.
        """
        filename = self.df['filename'].iloc[batch_idx]
        _, _, _, _, logits = outputs
        [logits] = detach_and_convert_tensors(pl_module, [logits])
        self.logits[filename] = logits

    def process_exams(self, model: LightningModule) -> None:
        """
        This method processes exams sequentially and makes predictions.

        Parameters
        ----------
            model : LightningModule     Lightning module instance used for inference.
        """
        # Compute predictions, logits, and probabilities for each exam
        predictions = []
        softmax_prob_pos = []
        softmax_prob_neg = []
        for _, row in self.exam_df.iterrows():
            instance_idcs = row['SelectedExamIndices']
            instance_rows = self.instances_df.iloc[instance_idcs]
            filenames = instance_rows['filename'].tolist()
            logits = [self.logits.get(filename, []) for filename in filenames]
            logits = np.vstack(logits)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            exam_probs = np.mean(probs, axis=0)
            softmax_prob_neg.append(exam_probs[0])
            softmax_prob_pos.append(exam_probs[1])
            predictions.append(np.argmax(exam_probs).item())

        # Ensure the DataFrame has the necessary columns
        self.exam_df['Predicted label'] = predictions
        self.exam_df[SOFTMAX_NEG] = softmax_prob_neg
        self.exam_df[SOFTMAX_POS] = softmax_prob_pos

    def generate_summary_plots(self) -> None:
        """
        This method creates teh ROC to be saved to local disk.
        """
        title = f"{self.dataset_name} Exam-Level ROC Curve"
        fig = plot_auroc_curve(df=self.exam_df, title=title, label_col=self.label_col)
        outpath = os.path.join(self.save_dir, title.replace(' ', '_') + '.png')
        fig.savefig(outpath)

    def log_to_wandb(self, trainer: Trainer) -> None:
        """
        This method logs the exam results to Weights & Biases (W&B).

        Parameters
        ----------
            trainer : Trainer     Trainer instance.
        """
        # Log the exam-level classification report and accuracy to W&B
        wb_logger = find_wandb_logger(trainer)
        if wb_logger:
            report = classification_report(self.exam_df[self.label_col],
                                           self.exam_df['Predicted label'],
                                           target_names=self.class_names,
                                           output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            outpath = os.path.join(
                self.save_dir, f"{self.dataset_name}_exam_classification_report.csv")
            report_df.to_csv(outpath, index=False)
            print(f"Classification report saved to: {outpath}")

            accuracy = accuracy_score(
                self.exam_df[self.label_col], self.exam_df['Predicted label'])
            print(f"Exam level accuracy: {accuracy}")
            wb_logger.log_hyperparams(
                {"Dataset": self.dataset_name, "Exam-level accuracy": accuracy})
            exam_table = wandb.Table(dataframe=self.exam_df)
            wandb.log({"Exam Results Table": exam_table})
            softmax_output = np.concatenate((
                np.expand_dims(self.exam_df[SOFTMAX_NEG].to_numpy(), axis=1),
                np.expand_dims(self.exam_df[SOFTMAX_POS].to_numpy(), axis=1)),
                axis=1)
            wandb.log({
                f"{self.dataset_name} Exam-level ROC curve": wandb.plot.roc_curve(
                    self.exam_df[self.label_col],
                    softmax_output,
                    labels=self.class_names,
                    title=f"{self.dataset_name} Exam-level ROC curve")
            })
