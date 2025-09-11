"""
Cnn2RnnClassifier.py

This is the definition of the Cnn2RnnClassifier architectures. It is
derived from the base class Cnn2RnnRegressor. This architectures enables the
configuration of CNN, RNN, and Regressor components separately via arguments
to the constructor, which may be specified in the Lightning CLI or via yaml
configuration files.

This architecture is designed to work with Dataset classes whose
batch output consists of (a) only the data and labels, or (b) data,
labels, video index, frame index via the verbose_data_module flag.

It is also designed to output either (a) just the final output, or (b)
the final output as well as intermediate values via the
report_intermediates flag.

Definitions:
    B   - batch_size
    L   - number of frames in video
    C   - input image channel count
    H   - input image height
    W   - input image width
    E   - embedding (feature vector) dimension
          when the CNN is cut after feature map collapse
    R   - RNN feature_dimension (context vector)
    CFM - channels of the CNN output feature map
          when CNN is cut at the feature map layer
    HFM - height feature dimension
          when CNN is cut at the feature map layer
    WFM - width feature dimension
          when CNN is cut at the feature map layer

Author: Olivia Zahn
        Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import warnings
from typing import Union, Tuple

import torch
from torch import nn, Tensor
from torchmetrics import Accuracy

from ghlobus.models.Cnn2RnnRegressor import Cnn2RnnRegressor
from ghlobus.models.TvCnnFeatureMap import TvCnnFeatureMap
from ghlobus.models.TvConvLSTM import TvConvLSTM
from ghlobus.models.MultiClassifier import MultiClassifier


class Cnn2RnnClassifier(Cnn2RnnRegressor):
    """
    Cnn2RnnClassifier implements a configurable deep learning architecture
    for video classification tasks, combining CNN, RNN, and classifier modules.

    This class allows flexible selection and configuration of the CNN, RNN,
    and classifier components via constructor arguments, supporting both
    standard and feature map-based CNNs, as well as various RNN types. It is
    designed to work with datasets that provide either simple (data, label)
    batches or verbose batches including indices, controlled by the
    `verbose_data_module` flag.

    The model supports reporting intermediate states (frame embeddings,
    context vectors, attention weights, logits) during inference via the
    `report_intermediates` flag, aiding interpretability and debugging.

    Attributes:
        cnn (nn.Module):              CNN component for feature extraction.
        rnn (nn.Module):              RNN component for temporal modeling.
        classifier (nn.Module):       Classifier module for final prediction.
        loss (nn.Module):             Loss function.
        activation (str|None):        Activation function for output.
        lr (float):                   Learning rate.
        num_classes (int):            Number of output classes.
        accuracy (Accuracy):          Torchmetrics accuracy metric.
        report_intermediates (bool):  Whether to report intermediate states.
        verbose_data_module (bool):   Whether to expect verbose batch format.

    Typical input shape: (B, L, C, H, W)
         where B=batch size, L=frames, C=channels, H=height, W=width.
    Output shape:        (B, num_classes) or tuple with intermediates if
         reporting is enabled.
    """

    def __init__(self,
                 cnn: nn.Module = TvCnnFeatureMap(),
                 rnn: nn.Module = TvConvLSTM(),
                 classifier: nn.Module = MultiClassifier(),
                 loss: nn.Module = nn.NLLLoss(),
                 activation: Union[str, None] = 'log_softmax',
                 lr: float = 5e-5,
                 report_intermediates=False,
                 ) -> None:
        """
        Initialize the Cnn2RnnClassifier model with the CNN, RNN, and Classifier components.

        Args:
            cnn: torch.nn.Module         - CNN component of the model
            rnn: torch.nn.Module         - RNN component of the model
            classifier: torch.nn.Module  - classifier component of the model
            loss: torch.nn.Module        - loss function to use
            activation: str or None      - activation function to use
            lr: float                    - learning rate for the optimizer
            report_intermediates: bool   - whether to return intermediate states
                                             in the forward method; intermediate states
                                             are used in inference to save frame embeddings,
                                             context vectors, and attention weights
            verbose_data_module: bool    - whether a verbose data module is used,
                                             which provides data samples and source indices,
                                             for example when there is randomness in sample
                                             selection; this is useful in inference to aid in
                                             interpretability of the model outputs.
        """
        # Initialize the base class
        # Ignore all warnings
        warnings.filterwarnings("ignore")
        super().__init__(cnn=cnn,
                         rnn=rnn,
                         loss=loss,
                         lr=lr,
                         report_intermediates=report_intermediates)

        # Record hyperparameters as attributes:
        self.loss = loss
        self.lr = lr

        # Configure the network components
        self.cnn = cnn
        self.rnn = rnn
        self.classifier = classifier
        if activation not in [None, 'log_softmax']:
            raise ValueError(f"Activation {activation} not recognized. "
                             f"Please use None or 'log_softmax'.")
        self.activation = activation

        # Number of classes
        self.num_classes = classifier.num_classes
        self.accuracy = Accuracy(task="multiclass",
                                 num_classes=self.num_classes)

        # Set the boolean flag for reporting intermediate states in the
        # forward loop; whether data_module is verbose or not
        self.report_intermediates = report_intermediates

        # Save the hyperparams in hparams file:
        self.save_hyperparameters(ignore=["cnn",
                                          "rnn",
                                          "classifier",
                                          "loss",
                                          "report_intermediates"])

    def forward(self, x: Tensor) -> \
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """
        Forward pass of data through the deep learning network.
        Input is 5D. (B, L, C, H, W)
        The following describes the dimensionality of the data pipeline at each step:
        - When CNN is TvCNN, converts from 5-dim to 3-dim. (B, L, E)
            or when CNN is TvCNNFeatureMap, from 5-dim to 4-dim  (B, L, CFM, HFM, WFM)
        - When RNN is LSTM or Attention, this will convert from 3-dim to 2-dim. (B, R)
            or when RNN is TvConvLSTM, from 5-dim to 2-dim. (B, R)
        - Classifier converts to final output. (B, num_classes)

        Parameters:
            x: torch.Tensor  - data to pass through the network. Expected input size is
                               (B, L, C, H, W) where B is batch size, L is number of
                               frames (subsampled),  C=number of channels in the image,
                               and H, W are image dimensions height and width.

        Returns:
            torch.Tensor:    - output of the network. Output size is (B, num_classes)
            or
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] - when report_intermediates is on.
        """
        # Step 1: CNN
        frame_features = self.cnn_forward(x)
        # Step 2: RNN
        context, auxiliary = self.rnn_forward(frame_features)
        # Step 3: Regressor
        y_hat, logits = self.classifier(context)

        # if self.report_intermediates, return output as well as intermediate values
        if self.report_intermediates:
            return y_hat, frame_features, context, auxiliary, logits
        else:
            # return only the final output
            return y_hat

    def cnn_forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the CNN model.

        Parameters:
            x: torch.Tensor     batch of concatenated videos
                                dims: (B, L, C, H, W)

        Returns:
            torch.Tensor:       batch of feature vectors
                                dims: (B, L, E) when plain CNN is used
                                dims: (B, L, fH, fW) when CnnFeatureMap is used
        """
        return self.cnn(x)

    def training_step(self, batch, batch_idx) -> float:
        """
        Training step. Passes batch data through model, computes loss, and reports loss
        using log_trainval_metrics function.

        Parameters:
            batch: tuple     - (x,y) pair of data (x) and label (y)
            batch_idx: int   - index value corresponding to (x,y) in the Dataset.

        Returns:
            loss: float      - Loss value computed using the model nn.Module Loss.
        """
        # Use _shared_eval_step to run data through the model and compute the loss
        loss, accuracy = self._shared_eval_step(batch, batch_idx)
        # Report the loss as train_loss metric
        metrics = {"train_loss": loss,
                   "train_accuracy": accuracy}
        self.log_trainval_metrics(metrics)
        # Return the loss
        return loss

    def validation_step(self, batch, batch_idx) -> float:
        """
        Validation step. Passes batch data through model, computes loss, and reports loss
        using log_trainval_metrics function.

        Parameters:
            batch: tuple       - (x,y) pair of data (x) and label (y)
            batch_idx: int     - index value corresponding to (x,y) in the Dataset.
        """
        loss, accuracy = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss,
                   "val_accuracy": accuracy}
        self.log_trainval_metrics(metrics)
        return loss

    def test_step(self, batch, batch_idx) -> float:
        """
        Test step. Passes batch data through model, computes loss, and reports loss
        using log_trainval_metrics function.

        Parameters:
            batch: tuple      - (x,y) pair of data (x) and label (y)
            batch_idx: int    - index value corresponding to (x,y) in the Dataset.
        """
        loss, accuracy = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss,
                   "test_accuracy": accuracy}
        self.log_dict(metrics)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for inference. Extracts input data from the batch and returns
        the model's output.

        Args:
            batch (tuple):          Batch of data, which may be (x, y), (uid, x, y), or
                                      other formats depending on the data module.
            batch_idx (int):        Index of the batch in the dataset.

        Returns:
            torch.Tensor or tuple:  Output of the model's forward pass. Shape is
                                      (B, num_classes) or a tuple with intermediates if
                                      `report_intermediates` is enabled.
        """
        # extract x from batch
        x = batch[0]

        return self.forward(x)

    def _shared_eval_step(self, batch, batch_idx) -> Tuple[float, float]:
        """
        Common evaluation step. Passes data through model then computes and return loss.

        - Logits are being converted to log softmax values.
        - Negative log-likelihood loss is applied to log softmax values.
        - Exp is applied to log softmax values to get back softmax probabilities.
        - Overall accuracy is calculated based on softmax probabilities and class labels (0,1).

        Parameters:
            batch: tuple        - (x,y) pair of data (x) and label (y)
            batch_idx: int      - index value corresponding to (x,y) in the Dataset (unused).

        Returns:                - 2-component tuple consisting of:
            loss: float         - Loss value computed using the model nn.Module Loss.
            accuracy: float     - Accuracy value computed using the model nn.Module Accuracy.
        """
        # extract x and y from batch or possibly also source indices
        x = batch[0]
        y = batch[1]

        # pass x through model
        y_hat = self.forward(x)

        # Unsqueeze y or y_hat if either is missing dimensionality (ie batch_size=1)
        if not y.shape:
            y = y.unsqueeze(-1)
        if not y_hat.shape:
            y_hat = y_hat.unsqueeze(-1)

        # Put y, y_hat on same device
        ind = y.get_device()
        device = "cpu" if ind == -1 else "cuda:" + str(ind)
        y_hat = y_hat.to(device)

        # NLLLoss expects long type
        y = y.long()
        # Compute loss on prediction
        loss = self.loss(y_hat, y)

        # Compute probabilities
        if self.activation is None:
            probs = y_hat
        elif self.activation.lower() == 'log_softmax':
            probs = torch.exp(y_hat)
        else:
            raise ValueError(f"Activation {self.activation} not recognized.")

        # Compute accuracy
        accuracy = self.accuracy(probs, y)

        # Return loss
        return loss, accuracy
