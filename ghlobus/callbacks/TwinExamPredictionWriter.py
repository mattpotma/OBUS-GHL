"""
TwinExamPredictionWriter.py

A class that tracks exam-level predictions and saves the results to disk.

Author: Daniel Shea
        Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import wandb
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Any, Tuple, Union, List
from sklearn.metrics import classification_report

from lightning.pytorch.callbacks import BasePredictionWriter
from lightning import Trainer, LightningModule

from ghlobus.utilities.inference_utils import detach_and_convert_tensors
from ghlobus.utilities.inference_utils import find_wandb_logger
from ghlobus.utilities.constants import SOFTMAX_NEG, SOFTMAX_POS
from ghlobus.utilities.constants import DEFAULT_VIDEO_PREDICTION_THRESHOLD
from ghlobus.utilities.plot_utils import show_mil_results


class TwinExamPredictionWriter(BasePredictionWriter):
    """
    A class used to write the output of the multiple gestation detection model.

    Attributes
    ----------
        save_dir : os.PathLike      Directory where the output will be saved.
        predictions : dict          Dictionary for tracking the prediction results.
    """

    def __init__(self,
                 save_dir: Union[os.PathLike, None] = None,
                 video_prediction_threshold: Union[float, None] = DEFAULT_VIDEO_PREDICTION_THRESHOLD,
                 save_plots: bool = False,
                 class_names: Union[List[str], None] = None,
                 label_col: Union[str, None] = None,
                 write_interval: str = "epoch",
                 ) -> None:
        """
        Constructs all the necessary attributes for the TWINPredictionWriter object.

        Parameters
        ----------
            save_dir : os.PathLike              Directory where the output will be saved.
            use_threshold : bool                Boolean indicating whether to use a threshold
                                                for the video-level predictions.
            video_prediction_threshold : float  Threshold value for the video-level
                                                predictions. Default is 0.45.
            save_plots : bool                   Boolean indicating whether summary plots should
                                                be generated with matplotlib and saved to the
                                                logging `save_dir` directory.
            class_names : list[str]             List of class names for classification tasks.
            label_col : str                     Column name for labels in the dataframe.
            write_interval : str                Interval at which the output will be written.
        """
        super().__init__(write_interval)
        self.save_dir = save_dir
        self.save_plots = save_plots
        self.predictions = {}
        self.logits = {}
        self.softmax_prob_pos = {}
        self.softmax_prob_neg = {}
        self.class_names = class_names
        self.label_col = label_col

        # Initialize the results DataFrame
        self.results = pd.DataFrame()
        self.dataset_name = None
        self.df = None

        # Thresholding
        self.video_prediction_threshold = video_prediction_threshold

        # Create the vector output directory and the predictions file.
        os.makedirs(self.save_dir, exist_ok=True)
        print(f'Created {self.save_dir} directory.')

    def setup(self,
              trainer: Trainer,
              pl_module: LightningModule,
              stage: str) -> None:
        """
        Sets up the necessary attributes for the class instance.

        This method extracts the dataset name and dataframe from the trainer's
        datamodule and assigns them to the instance.

        Parameters
        ----------
            trainer (Trainer):             Trainer instance containing the datamodule.
            pl_module (LightningModule):   LightningModule instance (not used in this
                                           method but can be used for other setups).
            stage (str):                   Stage of the process (not used in this method
                                           but can be used for different setups depending on
                                           the stage).

        Returns
        ----------
            None
        """
        # Extract the dataset name, otherwise raise error
        if hasattr(trainer.datamodule, 'dataset_name'):
            self.dataset_name = trainer.datamodule.dataset_name
        else:
            raise ValueError(f"Dataset name not found in trainer.datamodule.")

        # Extract the dataframe used in the dataset:
        self.df = trainer.datamodule.dfs[self.dataset_name]


    def on_predict_batch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule,
                             outputs: Any,
                             batch: Tuple,
                             batch_idx: int,
                             ) -> None:
        """
        Collects predictions at the end of each batch.

        Parameters
        ----------
            trainer : Trainer             PyTorch Lightning Trainer instance.
            pl_module : LightningModule   PyTorch Lightning Module instance.
            outputs : Any                 Output from the forward step.
            batch : Tuple                 Current batch of data.
            batch_idx : int               Index of the current batch.

        Returns
        ----------
            None
        """
        # Try using the batch indices to get exam info, which is required for building the results
        ds = trainer.predict_dataloaders.dataset
        all_exam_files, all_exam_inds, all_exam_tags = ds.get_exam(batch_idx)
        
        # ! Note - this code is assuming batch size of '1'
        # Extract the data from `batch` inputs and model `outputs`
        _, _, df_inds, _ = batch
        y_hat, _, _, _, logits = outputs
        
        # Detach and convert tensors to Numpy values.
        y_hat, logits, df_inds = detach_and_convert_tensors(pl_module, [y_hat, logits, df_inds])
        # Unpack y_hat, logits, and df_inds, since each is a list of tensors 
        # (list corresponds to elements of the batch, and batch_size is 1)
        y_hat = y_hat[0]
        logits = logits[0]
        df_inds = df_inds[0]
                
        # Access the DataFrame and retrieve a single data row from the DataFrame
        # ! Note - First [0] because we want the first element in batch (batch size of 1),
        #          Second [0] because we want the first element in the list
        vid0_idx = df_inds[0]
        vid0_row = self.df.iloc[vid0_idx]

        # Determine value for prediction
        if self.video_prediction_threshold is not None:
            # Threshold the video-level prediction
            # Must exponentiate bcs output is log-softmax!
            prediction = int(np.exp(y_hat)[1] > self.video_prediction_threshold)

        else:
            # Use the argmax of the softmax output as the prediction as default
            prediction = np.argmax(y_hat).item()
        
        # Prepare a data row to be appended to the results DataFrame
        exam_indices = np.unique(df_inds).tolist()
        data_row = {
            'exam_dir': vid0_row['exam_dir'],
            'StudyID': vid0_row['StudyID'],
            'GA': vid0_row['GA'],
            self.label_col: vid0_row[self.label_col],
            'Predicted label': prediction,
            SOFTMAX_NEG: np.exp(y_hat)[0],
            SOFTMAX_POS: np.exp(y_hat)[1],
            'Manufacturer': vid0_row['Manufacturer'],
            'ManufacturerModelName': vid0_row['ManufacturerModelName'],
            'Logits': logits,
            'ExamTagCombo': self.df.iloc[exam_indices]['tag'].tolist(),
            'TotalInstances': len(all_exam_files),
            'AllIndices': all_exam_inds,
            'AllTagsSet': set(all_exam_tags),
        }
        # Append the data row to the results DataFrame
        self.results = pd.concat([self.results, pd.DataFrame([data_row])], ignore_index=True)

    def on_predict_epoch_end(self,
                             trainer: Trainer,
                             pl_module: LightningModule,
                             ) -> None:
        """
        Modifies the dataframe, writes predictions to file at the end of each epoch, and
        logs results to weights and biases (if using).

        Parameters
        ----------
            trainer : Trainer             PyTorch Lightning Trainer instance.
            pl_module : LightningModule   PyTorch Lightning Module instance.

        Returns
        ----------
            None
        """
        # Compute the classification report
        report = classification_report(self.results[self.label_col],
                                       self.results['Predicted label'],
                                       target_names=self.class_names,
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Write the results dataframe
        outpath = os.path.join(self.save_dir, f"{self.dataset_name}_exam_predictions.csv")
        self.results.to_csv(outpath, index=False)
        print(f"Results saved to: {outpath}")

        # Write the classification report
        outpath = os.path.join(self.save_dir, f"{self.dataset_name}_exam_classification_report.csv")
        report_df.to_csv(outpath, index=False)
        print(f"Classification report saved to: {outpath}")

        # make `self.df` the results DataFrame, because next calls to downstream code 
        # expects it to be the results DataFrame
        self.df = self.results

        # If saving plots, generate and save them
        if self.save_plots:
            self.generate_summary_plots()

        # Write results to W&B, if needed
        self.log_to_wandb(trainer)

    def generate_summary_plots(self) -> None:
        """
        This method creates plots to be saved to local disk.

        Parameters
        ----------
            None. The method only requires access to `self.df` and `self.dataset_name` attribute.


        Returns
        ----------
            None. The method creates and writes figures to disk.
        """
        # Create ROC curve plot
        show_mil_results(
            experiment_name=self.dataset_name,
            labels=self.results[self.label_col].to_numpy(),
            scores=self.results[SOFTMAX_POS].to_numpy(),
            path=self.save_dir,
            attention_scores=None
        )
        

    def log_to_wandb(self, trainer: Trainer) -> None:
        """
        This method logs the video-level results to Weights & Biases (W&B).

        Parameters
        ----------
            trainer (Trainer):    PyTorch Lightning Trainer instance.

        Returns
        ----------
            None. The method logs the results to Weights & Biases.
        """
        wb_logger = find_wandb_logger(trainer)

        if wb_logger:
            # Record hyperparams for the dataset name and mean mae performance.
            wb_logger.log_hyperparams({"Dataset": self.dataset_name})

            # Record table to WandB
            wandb_table = wandb.Table(dataframe=self.results)

            # Log the table to the run.
            wandb.log({"Exam Results Table": wandb_table})

            softmax_output = np.concatenate(
                 (np.expand_dims(self.results[SOFTMAX_NEG].to_numpy(), axis=1),
                  np.expand_dims(self.results[SOFTMAX_POS].to_numpy(), axis=1)
                  ),
                 axis=1)

            # Get the ground truth labels
            ground_truth = self.results[self.label_col].to_numpy()

            # Log the ROC curve to the run.
            wandb.log({
                f"{self.dataset_name} Exam-level ROC curve": wandb.plot.roc_curve(
                    ground_truth,
                    softmax_output,
                    labels=self.class_names,
                    title=f"{self.dataset_name} Exam-level ROC curve",)
                 })
