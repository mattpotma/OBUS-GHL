"""
Cnn2RnnVectorWriter.py

A class that writes the intermediate results (frame embeddings, context vector,
and attention weights) to a hard disk.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
from typing import Tuple
import lightning as pl
from lightning.pytorch.callbacks import BasePredictionWriter
from ghlobus.utilities.inference_utils import save_intermediate_vectors


# noinspection PyUnresolvedReferences
class CnnVectorWriter(BasePredictionWriter):
    """
    A class used to write the output of the CNN2RNN model.

    Attributes
    ----------
        output_dir : os.PathLike   Directory where the output will be saved.
    """

    def __init__(self, save_dir: os.PathLike) -> None:
        """
        Constructs all the necessary attributes for the Cnn2RnnVectorWriter object.

        Parameters
        ----------
            save_dir : os.PathLike   Directory where the output will be saved.
            write_interval : str     Interval at which the output will be written.
        """
        super().__init__(write_interval="batch")
        self.output_dir = save_dir
        self.dataset_name = None
        self.df = None
        
    def setup(self,
              trainer: pl.Trainer,
              pl_module: pl.LightningModule,
              stage: str,
              ) -> None:
        """
        Extracts dataset name and dataframe from the datamodule.
        """
        self.dataset_name = trainer.datamodule.dataset_name
        self.df = trainer.datamodule.df

    def write_on_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           prediction: Tuple,
                           batch_indices: list,
                           batch: Tuple,
                           batch_idx: int,
                           dataloader_idx: int,
                           ) -> None:
        """
        Writes feature vectors, context vectors, and attention scores at the end of each batch.

        Parameters
        ----------
            trainer : pl.Trainer             PyTorch Lightning Trainer instance.
            pl_module : pl.LightningModule   PyTorch Lightning Module instance (ie the model).
            prediction : Tuple               Output from the forward step.
            batch_indices : list             Indices of the current batch.
            batch : Tuple                    Current batch of data.
            batch_idx : int                  Index of the current batch.
            dataloader_idx : int             Index of the current dataloader.

        Returns
        ----------
            None
        """
        # ! Note - this code is assuming batch size of '1'
        # Extract the data row and MediaStorageSOPInstanceUID (unique identifier)
        filename = self.df['filename'].iloc[batch_idx]
        # Pull out the relevant features intended to be written:
        _, frame_features, context_vectors, attention_scores = prediction
        # Write out the frame_features, context vector, and attention vectors for the video
        save_intermediate_vectors(output_dir=self.output_dir,
                                  sample_id=filename,
                                  frame_features=frame_features,
                                  context_vectors=context_vectors,
                                  attention_scores=attention_scores)
