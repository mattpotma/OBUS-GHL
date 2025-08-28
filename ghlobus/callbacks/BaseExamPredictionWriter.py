"""
BaseExamPredictionWriter.py

A base class for exam-level analysis prediction writers.

Author: Daniel Shea
        Olivia Zahn

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Union

from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter

from ghlobus.utilities.exam_methods import build_exams_df
from ghlobus.utilities.exam_methods import define_exam


# noinspection PyUnresolvedReferences,PyMethodOverriding
class BaseExamPredictionWriter(BasePredictionWriter, ABC):
    """
    A base class for exam-level analysis prediction writers.

    This class provides a common structure for accumulating instance-level predictions
    and performing exam-level analysis. It defines a common interface for setup,
    processing, and logging, with abstract methods for model-specific logic.

    Attributes
    ----------
        save_dir : os.PathLike                 Directory where the output will be saved.
        max_instances_per_exam : None or int   Upper limit of the number of videos per exam to consider.
        use_known_tag_combos : bool            Whether to use the known_tag_combos list.
        default_exam_cols : List[str]          Default columns defining an exam.
        columns_to_record : List[str] or Dict  Columns to record in the exam-level predictions.
        save_plots : bool                      Boolean indicating whether summary plots should be generated.
        dataset_name : str                     Name of the dataset being processed.
        instances_df : pd.DataFrame            DataFrame containing instance-level data.
        exam_df : pd.DataFrame                 DataFrame containing exam-level data.
    """

    def __init__(self,
                 save_dir: os.PathLike,
                 max_instances_per_exam: Union[int, None] = None,
                 use_known_tag_combos: bool = False,
                 default_exam_cols: Union[List[str], None] = None,
                 columns_to_record: Union[List[str], None] = None,
                 save_plots: bool = False,
                 class_names: Union[List[str], None] = None,
                 label_col: Union[str, None] = None,
                 feature_name: Union[str, None] = None,
                 write_interval: str = "epoch") -> None:
        """
        Constructs all the necessary attributes for the BaseExamPredictionWriter object.

        Parameters
        ----------
            save_dir : os.PathLike                 Directory where the output will be saved.
            max_instances_per_exam : None or int   Upper limit of the number of videos per exam to
                                                   consider. Set to None to consider all videos.
            use_known_tag_combos : bool            Whether to use the known_tag_combos list.
            default_exam_cols : List[str]          Default columns defining an exam.
            columns_to_record : List[str] or Dict  Columns to record in the exam-level predictions.
            save_plots : bool                      Boolean indicating whether summary plots should be generated.
            class_names : list[str]                List of class names for classification tasks.
            label_col : str                        Column name for labels in the dataframe.
            feature_name : str                     Name of the feature being predicted (optional).
            write_interval : str                   Interval at which the output will be written.
        """
        super().__init__(write_interval)
        self.save_dir = save_dir
        self.max_instances_per_exam = max_instances_per_exam
        self.use_known_tag_combos = use_known_tag_combos
        self.default_exam_cols = default_exam_cols
        self.columns_to_record = columns_to_record
        self.save_plots = save_plots
        self.class_names = class_names
        self.label_col = label_col
        self.feature_name = feature_name
        self.dataset_name = None
        self.instances_df = None
        self.exam_df = None
        self.df = None

    def setup(self,
              trainer: Trainer,
              pl_module: LightningModule,
              stage: str,
              ) -> None:
        """
        This method sets up the necessary dataframes and directories for processing exams.

        Parameters
        ----------
            trainer (Trainer):            PyTorch Lightning Trainer instance.
            pl_module (LightningModule):  PyTorch Lightning Module instance.
            stage (str):                  Stage of the process.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        print(f'Created {self.save_dir} directory.')
        self.dataset_name = trainer.datamodule.dataset_name
        self.df = self.instances_df = trainer.datamodule.df

        self.exam_df = build_exams_df(
            df=self.df,
            columns_defining_exam=self.default_exam_cols,
            known_tag_combos=self.known_tag_combos,
            columns_to_record=self.columns_to_record)
        self.exam_df['SelectedExamIndices'] = self.exam_df.apply(
            lambda x: define_exam(x, self.df, max_videos_per_exam=self.max_instances_per_exam), axis=1)
        self.exam_df.reset_index(drop=True, inplace=True)

    @abstractmethod
    def on_predict_batch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule,
                             outputs: Any,
                             batch: Tuple,
                             batch_idx: int) -> None:
        """
        This method logs the predictions for each video in the batch.
        """
        raise NotImplementedError

    def on_predict_epoch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule) -> None:
        """
        Process each exam using the model, then write the predictions to file.
        Also engages logging to W&B, if a WandB logger is being used.

        Parameters
        ----------
            trainer (Trainer):            PyTorch Lightning Trainer instance.
            pl_module (LightningModule):  PyTorch Lightning Module instance.
        """
        self.process_exams(pl_module)
        outpath = os.path.join(
            self.save_dir, f"{self.dataset_name}_exam_predictions.csv")
        self.exam_df.to_csv(outpath, index=False)
        print(f"Results saved to: {outpath}")

        if self.save_plots:
            self.generate_summary_plots()

        self.log_to_wandb(trainer)

    @abstractmethod
    def process_exams(self, model: LightningModule) -> None:
        """
        This method processes exams sequentially and makes predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_summary_plots(self) -> None:
        """
        This method creates plots to be saved to local disk.
        """
        raise NotImplementedError

    @abstractmethod
    def log_to_wandb(self, trainer: Trainer) -> None:
        """
        This method logs the exam results to Weights & Biases (W&B).
        """
        raise NotImplementedError
