"""
BaseVideoPredictionWriter.py

An abstract base class for writing video-level model predictions to disk.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Union

from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter


# noinspection PyUnresolvedReferences
class BaseVideoPredictionWriter(BasePredictionWriter, ABC):
    """
    An abstract base class for writing video-level model predictions.

    This class provides a common structure for collecting predictions, saving them
    to a file, and logging results. Child classes must implement the abstract
    methods to handle model-specific logic.

    Attributes
    ----------
        save_dir : os.PathLike   Directory where the output will be saved.
        save_plots : bool        Whether to generate and save summary plots.
        predictions : dict       Dictionary for tracking prediction results.
        dataset_name : str       Name of the dataset being processed.
        df : pd.DataFrame        Dataframe associated with the dataset.
    """

    def __init__(self,
                 save_dir: os.PathLike,
                 save_plots: bool = False,
                 class_names: Union[List[str], None] = None,
                 label_col: Union[str, None] = None,
                 feature_name: Union[str, None] = None,
                 write_interval: str = "epoch") -> None:
        """
        Constructs the BaseVideoPredictionWriter.

        Parameters
        ----------
            save_dir : os.PathLike   Directory to save output.
            save_plots : bool        If True, generate and save summary plots.
            class_names : list[str]  List of class names for classification tasks.
            label_col : str          Column name for labels in the dataframe.
            feature_name : str       Name of the feature being predicted (optional).
            write_interval : str     When to write predictions ('batch' or 'epoch').
        """
        super().__init__(write_interval)
        self.save_dir = save_dir
        self.save_plots = save_plots
        self.predictions = {}
        self.dataset_name = None
        self.class_names = class_names
        self.label_col = label_col
        self.feature_name = feature_name
        self.df = None

        os.makedirs(self.save_dir, exist_ok=True)
        print(f'Created {self.save_dir} directory.')

    def setup(self,
              trainer: Trainer,
              pl_module: LightningModule,
              stage: str,
              ) -> None:
        """
        Extracts dataset name and dataframe from the datamodule.

        Parameters
        ----------
            trainer : Trainer            Trainer instance.
            pl_module : LightningModule  LightningModule instance.
            stage : str                  Stage of the training process
                                         (e.g., 'fit', 'predict', 'test').
        """
        self.dataset_name = trainer.datamodule.dataset_name
        self.df = trainer.datamodule.df

    @abstractmethod
    def on_predict_batch_end(self, trainer: Trainer,
                             pl_module: LightningModule,
                             outputs: Any,
                             batch: Tuple,
                             batch_idx: int,
                             dataloader_idx: int = 0,
                             ) -> None:
        """
        Abstract method to collect predictions at the end of each batch.
        Must be implemented by child classes.
        """
        raise NotImplementedError

    def on_predict_epoch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule,
                             ) -> None:
        """
        Writes predictions to file and logs results at the end of the epoch.

        Parameters
        ----------
            trainer : Trainer            Trainer instance.
            pl_module : LightningModule  LightningModule instance.
        """
        # Process and save predictions (implemented in child)
        self._process_and_save_predictions()

        # Write the main dataframe
        outpath = os.path.join(
            self.save_dir, f"{self.dataset_name}_video_predictions.csv")
        self.df.to_csv(outpath, index=False)
        print(f"Results saved to: {outpath}")

        # Generate plots if requested
        if self.save_plots:
            self.generate_summary_plots()

        # Log to W&B if a logger is present
        self.log_to_wandb(trainer)

    @abstractmethod
    def _process_and_save_predictions(self) -> None:
        """
        Abstract method to process collected predictions and add them to the dataframe.
        This is where model-specific columns and metrics are computed.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_summary_plots(self) -> None:
        """
        Abstract method to generate and save model-specific summary plots.
        """
        raise NotImplementedError

    @abstractmethod
    def log_to_wandb(self, trainer: Trainer) -> None:
        """
        Abstract method to log model-specific results to Weights & Biases.
        """
        raise NotImplementedError
