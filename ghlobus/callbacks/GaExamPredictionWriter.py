"""
GaExamPredictionWriter.py

A class used to perform exam-level gestational age analysis.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
from typing import Any, Tuple, List, Union

import torch
from lightning import Trainer, LightningModule

from ghlobus.callbacks.RegressorExamPredictionWriter import RegressorExamPredictionWriter
from ghlobus.utilities.inference_utils import detach_and_convert_tensors
from ghlobus.utilities.inference_utils import get_rescale_log_value_func
from ghlobus.utilities.inference_utils import LGA_MEAN, LGA_STD
from ghlobus.utilities.sweep_utils import DEFAULT_EXAM_COLS
from ghlobus.utilities.sweep_utils import KNOWN_COMBO_TAGS
from ghlobus.utilities.sweep_utils import COLUMNS_TO_RECORD


class GaExamPredictionWriter(RegressorExamPredictionWriter):
    """
    Performs exam-level gestational age analysis by aggregating frame-level features,
    predicting gestational age using a deep learning model, and storing results in
    exam dataframes. Supports configurable exam columns, tag combinations, and output
    formats. Extends `RegressorExamPredictionWriter` with gestational age-specific
    logic and error calculation.
    """
    def __init__(self,
                 save_dir: os.PathLike,
                 max_instances_per_exam: Union[int, None] = None,
                 use_known_tag_combos: bool = False,
                 default_exam_cols: List[str] = DEFAULT_EXAM_COLS,
                 columns_to_record: List[str] = COLUMNS_TO_RECORD['GA'],
                 save_plots: bool = False,
                 write_interval: str = "epoch") -> None:
        """
        Initializes the GaExamPredictionWriter for exam-level gestational age analysis.

        Args:
            save_dir (os.PathLike):         Directory to save output files and results.
            max_instances_per_exam (int):   Maximum number of instances to use per exam. Defaults to None.
            use_known_tag_combos (bool):    Whether to use predefined tag combinations for exams.
                                              Defaults to False.
            default_exam_cols (List[str]):  List of default columns to include in exam dataframes.
                                              Defaults to DEFAULT_EXAM_COLS.
            columns_to_record (List[str]):  List of columns to record for gestational age analysis.
                                              Defaults to COLUMNS_TO_RECORD['GA'].
            save_plots (bool):              Whether to save plots during analysis.
                                              Defaults to False.
            write_interval (str):           Interval for writing results (e.g., "epoch").
                                              Defaults to "epoch".
        """
        super().__init__(
            label_name='ga_boe',
            label_plot_name='GA',
            unit_plot_name='Days',
            use_percentile_bland_altman=False,
            include_trimester_plots=True,
            save_dir=save_dir,
            max_instances_per_exam=max_instances_per_exam,
            use_known_tag_combos=use_known_tag_combos,
            default_exam_cols=default_exam_cols,
            columns_to_record=columns_to_record,
            save_plots=save_plots,
            write_interval=write_interval
        )
        self.known_tag_combos = KNOWN_COMBO_TAGS if use_known_tag_combos else None

    def on_predict_batch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule,
                             outputs: Any,
                             batch: Tuple,
                             batch_idx: int) -> None:
        """
        Called at the end of each prediction batch during inference.

        Extracts frame-level features from the model outputs, detaches and converts them,
        and stores them in the `frame_features` dictionary using the corresponding filename
        from the batch.

        Args:
            trainer (Trainer):            PyTorch Lightning Trainer instance.
            pl_module (LightningModule):  LightningModule being used for prediction.
            outputs (Any):                Outputs returned by the model for the batch.
            batch (Tuple):                Input batch data.
            batch_idx (int):              Index of the current batch.
        """
        filename = self.df['filename'].iloc[batch_idx]
        frame_features = outputs[1]
        [frame_features] = detach_and_convert_tensors(
            pl_module, [frame_features])
        self.frame_features[filename] = frame_features

    def process_exams(self, model: LightningModule) -> None:
        """
        Processes all exams by extracting frame-level features, predicting gestational
        age using the provided model, rescales the predictions from log scale to days,
        and stores both log and scaled predictions in the exam DataFrame. After
        processing, calculates prediction errors using the base class method.

        Args:
            model (LightningModule):  Trained model used for prediction.
        """
        zlog_predictions = []
        scaled_predictions = []

        # Create a rescale function for log GA
        print(f"Using hard-coded log_GA mean {LGA_MEAN} and std {LGA_STD}.")
        rescale_log_ga = get_rescale_log_value_func(mean=LGA_MEAN, std=LGA_STD)

        # Analyze each exam
        for _, row in self.exam_df.iterrows():
            instance_idcs = row['SelectedExamIndices']
            instance_rows = self.instances_df.iloc[instance_idcs]
            filenames = instance_rows['filename'].tolist()
            frame_features = [self.frame_features.get(
                filename, []) for filename in filenames]
            frame_features = [torch.from_numpy(x).squeeze(0).to(
                dtype=torch.float32) for x in frame_features if len(x) > 0]
            if not frame_features:
                zlog_predictions.append(0)
                scaled_predictions.append(0)
                continue
            frame_features = torch.vstack(frame_features)
            y_hat = self.analyze_exam(model, frame_features)
            zlog_predictions.append(y_hat)
            # Convert log GA to days for storage
            predicted_ga_days = rescale_log_ga(y_hat)
            scaled_predictions.append(predicted_ga_days)

        # Store both log and scaled predictions
        self.exam_df['Predicted Z Log GA'] = zlog_predictions
        self.exam_df[self.predicted_col_name] = scaled_predictions

        # Use base class error calculation
        self._process_and_calculate_errors()

    def analyze_exam(self, model: Any,
                     frame_features: torch.Tensor) -> float:
        """
        Performs gestational age prediction for a single exam using the provided
        model and frame-level features.

        Args:
            model (Any):                   Trained model with `rnn_forward` and
                                             `regressor` methods.
            frame_features (torch.Tensor): Tensor containing frame-level features
                                             for the exam.

        Returns:
            float: Predicted log gestational age for the exam.
        """
        x = frame_features
        x = x.to(device=model.device)
        x = torch.unsqueeze(x, dim=0)
        y_hat, _ = model.rnn_forward(x)
        y_hat = model.regressor(y_hat)
        y_hat = y_hat.squeeze().detach().cpu().item()
        return y_hat
