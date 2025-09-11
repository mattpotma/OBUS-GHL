"""
EfwVideoPredictionWriter.py

A class that tracks video-level EFW predictions and saves the results to disk.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from typing import Any, Tuple
from lightning import Trainer, LightningModule

from ghlobus.callbacks.RegressorVideoPredictionWriter import RegressorVideoPredictionWriter
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


class EfwVideoPredictionWriter(RegressorVideoPredictionWriter):
    """
    Writes predictions for the Estimated Fetal Weight (EFW) regression model.
    """

    def __init__(self, *args, **kwargs):
        # Hard-code EFW-specific configuration
        super().__init__(
            label_name='EFW',
            label_plot_name='EFW',
            unit_plot_name='g',
            use_percentile_bland_altman=True,  # EFW uses percentile Bland-Altman plot
            include_trimester_plots=False,     # EFW doesn't include trimester plots
            *args, **kwargs
        )

        self.sample_rescaler = BioNormLogRescale()
        self.biometrics = {
            'AC': [],
            'FL': [],
            'HC': [],
            'BPD': [],
            'zlog_ac': [],
            'zlog_fl': [],
            'zlog_hc': [],
            'zlog_bpd': [],
        }

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
                             batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> None:
        # filename can be accessed from the dataframe using batch_idx, since batch size is 1
        filename = self.df['filename'].iloc[batch_idx]
        y_hat = outputs[0]
        [y_hat] = detach_and_convert_tensors(pl_module, [y_hat])
        # Unpack biometric measurements from y_hat
        zlog_ac, zlog_fl, zlog_hc, zlog_bpd = y_hat
        # Store the predicted EFW in the dictionary
        self.predictions[filename] = self.sample_rescaler.apply(y_hat)
        # Add the biometric measurements to their respective lists
        self.biometrics['AC'].append(
            rescale_log_value(zlog_ac, AC_MEAN, AC_STD))
        self.biometrics['FL'].append(
            rescale_log_value(zlog_fl, FL_MEAN, FL_STD))
        self.biometrics['HC'].append(
            rescale_log_value(zlog_hc, HC_MEAN, HC_STD))
        self.biometrics['BPD'].append(
            rescale_log_value(zlog_bpd, BPD_MEAN, BPD_STD))
        self.biometrics['zlog_ac'].append(zlog_ac.item())
        self.biometrics['zlog_fl'].append(zlog_fl.item())
        self.biometrics['zlog_hc'].append(zlog_hc.item())
        self.biometrics['zlog_bpd'].append(zlog_bpd.item())

    def _add_custom_processing(self) -> None:
        """
        Add EFW-specific processing for biometric measurements.
        """
        # Add biometric measurements to the dataframe
        for biometric_name, values in self.biometrics.items():
            if len(values) == len(self.df):
                self.df[biometric_name] = values
            else:
                print(
                    f"Warning: Biometric {biometric_name} has {len(values)} values but dataframe has {len(self.df)} rows")
