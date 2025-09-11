"""
EfwExamPredictionWriter.py

A class used to perform exam-level estimated fetal weight analysis.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
from typing import Any, Tuple, List, Union
from collections import defaultdict

import torch
from lightning import Trainer, LightningModule

from ghlobus.callbacks.RegressorExamPredictionWriter import RegressorExamPredictionWriter
from ghlobus.utilities.inference_utils import detach_and_convert_tensors
from ghlobus.utilities.inference_utils import BioNormLogRescale
from ghlobus.utilities.inference_utils import rescale_log_value
from ghlobus.utilities.inference_utils import AC_MEAN
from ghlobus.utilities.inference_utils import AC_STD
from ghlobus.utilities.inference_utils import HC_MEAN
from ghlobus.utilities.inference_utils import HC_STD
from ghlobus.utilities.inference_utils import BPD_MEAN
from ghlobus.utilities.inference_utils import BPD_STD
from ghlobus.utilities.inference_utils import FL_MEAN
from ghlobus.utilities.inference_utils import FL_STD
from ghlobus.utilities.sweep_utils import DEFAULT_EXAM_COLS
from ghlobus.utilities.sweep_utils import KNOWN_COMBO_TAGS
from ghlobus.utilities.sweep_utils import COLUMNS_TO_RECORD


class EfwExamPredictionWriter(RegressorExamPredictionWriter):
    """
    A class used to perform exam-level gestational age analysis.
    """

    def __init__(self,
                 save_dir: os.PathLike,
                 max_instances_per_exam: Union[int, None] = None,
                 use_known_tag_combos: bool = False,
                 default_exam_cols: List[str] = DEFAULT_EXAM_COLS,
                 columns_to_record: List[str] = COLUMNS_TO_RECORD["EFW"],
                 save_plots: bool = False,
                 write_interval: str = "epoch") -> None:
        super().__init__(
            label_name='EFW',
            label_plot_name='EFW',
            unit_plot_name='g',
            use_percentile_bland_altman=True,
            include_trimester_plots=False,
            save_dir=save_dir,
            max_instances_per_exam=max_instances_per_exam,
            use_known_tag_combos=use_known_tag_combos,
            default_exam_cols=default_exam_cols,
            columns_to_record=columns_to_record,
            save_plots=save_plots,
            write_interval=write_interval
        )
        self.known_tag_combos = KNOWN_COMBO_TAGS if use_known_tag_combos else None
        self.sample_rescaler = BioNormLogRescale()

    def on_predict_batch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule,
                             outputs: Any,
                             batch: Tuple,
                             batch_idx: int) -> None:
        filename = self.df['filename'].iloc[batch_idx]
        frame_features = outputs[1]
        [frame_features] = detach_and_convert_tensors(
            pl_module, [frame_features])
        self.frame_features[filename] = frame_features

    def process_exams(self, model: LightningModule) -> None:
        predictions = []
        biometrics = defaultdict(list)
        for _, row in self.exam_df.iterrows():
            instance_idcs = row['SelectedExamIndices']
            instance_rows = self.instances_df.iloc[instance_idcs]
            filenames = instance_rows['filename'].tolist()
            frame_features = [self.frame_features.get(
                filename, []) for filename in filenames]
            frame_features = [torch.from_numpy(x).squeeze(0).to(
                dtype=torch.float32) for x in frame_features if len(x) > 0]
            if not frame_features:
                predictions.append(0)
                continue
            frame_features = torch.vstack(frame_features)
            outputs = self.analyze_exam(model, frame_features)
            y_hat = list(outputs[0])
            zlog_ac, zlog_fl, zlog_hc, zlog_bpd = y_hat
            # Re-scale the values and apply the hadlock formula
            predicted_efw = self.sample_rescaler.apply(y_hat)
            predictions.append(predicted_efw)
            # Add the biometric measurements to their respective lists
            biometrics['Predicted AC'].append(
                rescale_log_value(zlog_ac, AC_MEAN, AC_STD))
            biometrics['Predicted FL'].append(
                rescale_log_value(zlog_fl, FL_MEAN, FL_STD))
            biometrics['Predicted HC'].append(
                rescale_log_value(zlog_hc, HC_MEAN, HC_STD))
            biometrics['Predicted BPD'].append(
                rescale_log_value(zlog_bpd, BPD_MEAN, BPD_STD))
            biometrics['Predicted zlog_ac'].append(zlog_ac)
            biometrics['Predicted zlog_fl'].append(zlog_fl)
            biometrics['Predicted zlog_hc'].append(zlog_hc)
            biometrics['Predicted zlog_bpd'].append(zlog_bpd)

        # Store predictions and add biometric columns
        self.exam_df[self.predicted_col_name] = predictions
        for key, values in biometrics.items():
            self.exam_df[key] = values

        # Use base class error calculation
        self._process_and_calculate_errors()

    def analyze_exam(self, model: Any, frame_features: torch.Tensor) -> Any:
        x = frame_features
        x = x.to(device=model.device)
        x = torch.unsqueeze(x, dim=0)
        y_hat, _ = model.rnn_forward(x)
        y_hat = model.regressor(y_hat)
        [y_hat] = detach_and_convert_tensors(model, [y_hat])
        return y_hat
