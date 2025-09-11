"""
Cnn2RnnRegressor.py

This is the definition of the Cnn2RnnRegressor architectures and enables
configuration of CNN, RNN, and Regressor components separately via arguments
to the constructor, which may be specified in the Lightning CLI or via yaml
configuration files.

This architecture is designed to work with Dataset classes whose
batch output consists of only the data and labels.

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
    R   - RNN feature dimension (context vector)

Author: Daniel Shea
        Olivia Zahn
        Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
from torch import nn, optim, Tensor
from lightning import LightningModule

from ghlobus.models.TvCnn import TvCnn
from ghlobus.models.BasicAdditiveAttention import BasicAdditiveAttention


class Cnn2RnnRegressor(LightningModule):
    """
    Cnn2RnnRegressor implements a configurable deep learning architecture for
    video regression tasks, combining CNN, RNN, and regressor modules.

    This class allows flexible selection and configuration of the CNN, RNN,
    and regressor components via constructor arguments, supporting both standard
    and attention-based RNNs. It is designed to work with datasets that provide
    (data, label) batches.

    The model supports reporting intermediate states (frame embeddings, context
    vectors, attention weights) during inference via the `report_intermediates`
    flag, aiding interpretability and debugging.

    Attributes:
        cnn (nn.Module):         CNN component for feature extraction.
        rnn (nn.Module):         RNN component for temporal modeling.
        regressor (nn.Module):   Regressor module for final prediction.
        loss (nn.Module):        Loss function.
        lr (float):              Learning rate.
        report_intermediates (bool): Whether to report intermediate states.

    Typical input shape: (B, L, C, H, W)
        where B=batch size, L=frames, C=channels, H=height, W=width.
    Output shape:        (B,) or tuple with intermediates if reporting is
        enabled.
    """

    def __init__(self,
                 cnn: nn.Module = TvCnn(),
                 rnn: nn.Module = BasicAdditiveAttention(),
                 regressor: nn.Module = nn.Linear(
                     in_features=1000, out_features=1),
                 loss: nn.Module = nn.L1Loss(reduction='mean'),
                 lr: float = 5e-5,
                 report_intermediates: bool = False,
                 ):
        """
        Parameters:
            cnn: nn.Module        - CNN vision network. Converts Tensor of dimensions
                                    (B*L,C,H,W) to stack of tensors (B*L, E) where
                                    E = length of feature vector output by CNN.
            rnn: nn.Module        - Torch nn.Module for RNN-based module. It should take a sequence
                                    of vectors and convert it to a single vector. Input dimensions
                                    (B*L, L) and output dimensions of (B, R). Aggregates
                                    information over the 1-dimension (Frames/temporal dim).
            regressor: nn.Module  - Module to project the output vector from the RNN to a single
                                    numeric value. Input size (B, R) and output dimension (B,1).
            loss: nn.Module       - Loss for training.
            lr: float             - learning rate (! NOTE: OVERRIDDEN BY LIGHTNING-CLI IF `optimizer`
                                    SPECIFIED.)
            report_intermediates  - boolean indicating if the .forward() method should return
                                    intermediate vectors (frame encoder vectors, attention vectors,
                                    context vectors) with the final prediction.

        Note: Subclasses overriding this method should call super().__init__(), log parameters
            with save_hyperparameters, and MUST configure a self.cnn, self.rnn, and self.regressor.
            attributes representing the network.
        """
        # Initialize LightningModule
        super().__init__()

        # Record hyperparameters as attributes:
        self.loss = loss
        self.lr = lr

        # Configure the network components
        self.cnn = cnn
        self.rnn = rnn
        self.regressor = regressor

        # Set the boolean flag for reporting intermediate states in the
        # forward loop
        self.report_intermediates = report_intermediates

        # Save the hyperparams in hparams file:
        self.save_hyperparameters(ignore=["cnn",
                                          "rnn",
                                          "regressor",
                                          "loss",
                                          "report_intermediates"])

    def forward(self, x: Tensor):
        """
        Forward pass of data through the deep learning network.
        CNN will convert data from 5-dim to 3-dim. (Batch, Spatial+Color+Temporal)
        RNN will convert from 3-dim to 2-dim. (Batch, ContextVector)
        Regressor converts to final output. (Batch, num outputs)

        Parameters:
            x: torch.Tensor  - data to pass through the network. Expected input size is
                               (B, L, C, H, W) where B is batch size, L is number of
                               frames subsampled,  C=number of channels in the image,
                               and H,W are image dimensions height and width.
        """
        # Step 1: CNN
        frame_features = self.cnn_forward(x)
        # Step 2: RNN
        context, attention = self.rnn_forward(frame_features)
        # Step 3: Regressor
        y_hat = self.regressor(context)
        # squeeze to remove extra dimension that regressors often add
        # typically no downside if it can't be squeezed ;)
        y_hat = y_hat.squeeze()
        # if self.report_intermediates, return all
        if self.report_intermediates:
            return y_hat, frame_features, context, attention
        return y_hat

    def cnn_forward(self, x: Tensor):
        """
        Forward pass of the CNN model.

        Parameters:
            x: torch.Tensor     batch of concatenated videos
                                dims: (B, L, C, H, W)

        Returns:
            torch.Tensor:       batch of feature vectors
                                dims: (B, L, E)
        """
        # STEP 1: FEATURIZE INPUT FRAMES
        # Pass each frame in x through CNN feature extractor:
        return self.cnn(x)

    def rnn_forward(self, x: Tensor):
        """
        Forward pass of the RNN model.

        Parameters:
            x (torch.Tensor):    batch of concatenated videos
                                 dims: (B, L, E)
        Returns:
            torch.Tensor:       batch of feature vectors
                                dims: (B, 1, R)
        """
        # STEP 2: TEMPORALLY AGGREGATE frame feature vectors
        return self.rnn(x)

    def log_trainval_metrics(self, metrics: dict):
        """
        Logs training time metrics from train DataLoader and validation DataLoader.

        Parameters:
            metrics: dictionary    metrics to log to Lightning Logger (by default, TensorBoard)
        """
        # Note the frequency of metrics reported (epoch, not step)
        # prog_bar: print on progress bar
        # sync_dist: for syncing output across distributed training
        kwargs = {
            'on_step': False,
            'on_epoch': True,
            'prog_bar': True,
            'sync_dist': True,
        }
        self.log_dict(metrics, **kwargs)

    def training_step(self, batch, batch_idx):
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
        loss = self._shared_eval_step(batch, batch_idx)
        # Report the loss as train_loss metric
        metrics = {"train_loss": loss}
        self.log_trainval_metrics(metrics)
        # Return the loss
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step. Passes batch data through model, computes loss, and reports loss
        using log_trainval_metrics function.

        Parameters:
            batch: tuple       - (x,y) pair of data (x) and label (y)
            batch_idx: int     - index value corresponding to (x,y) in the Dataset.
        """
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss}
        self.log_trainval_metrics(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step. Passes batch data through model, computes loss, and reports loss
        using log_trainval_metrics function.

        Parameters:
            batch: tuple      - (x,y) pair of data (x) and label (y)
            batch_idx: int    - index value corresponding to (x,y) in the Dataset.
        """
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss}
        self.log_dict(metrics)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Prediction step. Passes batch data through model, and returns result.

        Parameters:
            batch: tuple     - (x,y) pair of data (x) and label (y)
            batch_idx: int   - index value corresponding to (x,y) in the Dataset.
        """
        # extract x and y from batch
        x = batch[0]
        # Return self.forward(x)
        return self.forward(x)

    # noinspection PyUnusedLocal
    def _shared_eval_step(self, batch, batch_idx):
        """
        Common evaluation step. Passes data through model then computes and return loss.

        Parameters:
            batch: tuple        - (x,y) pair of data (x) and label (y)
            batch_idx: int      - index value corresponding to (x,y) in the Dataset.

        Returns:
            loss: float         - Loss value computed using the model nn.Module Loss.
        """
        # extract x and y from batch
        x, y = batch
        # pass x through model
        y_hat = self.forward(x)
        # Unsqueeze y or y_hat if either is missing dimensionality (ie batch_size=1)
        if not y.shape:
            y = y.unsqueeze(-1)
        if not y_hat.shape:
            y_hat = y_hat.unsqueeze(-1)
        # Compute loss on prediction
        loss = self.loss(y_hat, y)
        # Return loss
        return loss

    def configure_optimizers(self):
        """
        Configures and returns the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer instance with model parameters and specified hyperparameters.
        """
        # Using the 'standardized' settings from additive attention experiments
        opt = optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1.0e-08,
            weight_decay=0.0,
            amsgrad=False,
            foreach=None,
            maximize=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        return opt
