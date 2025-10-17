"""Export to TFLite."""

import argparse
from pathlib import Path
import tensorflow as tf
import torch
import torch.nn as nn

import ai_edge_torch
from ghlobus.utilities.inference_utils import (
    get_dicom_frames,
    load_Cnn2RnnRegressor_model,
    load_Cnn2RnnClassifier_model,
    enumerate_dicom_files,
    video_level_inference_FP,
    exam_level_inference_FP,
    create_output_directories,
    write_results,
    LGA_MEAN,
    LGA_STD,
)
from ghlobus.models.BasicAdditiveAttention import BasicAdditiveAttention

# Set default_model and default_dicom
DEFAULT_DEVICE = "cpu"
DEFAULT_OUTDIR = "./test_output/"
ALLOWED_TASKS = ["GA", "GA_FRAMEWISE", "FP"]


class ModelWithRescaling(nn.Module):
    """Wrapper that adds log->days rescaling to model output."""

    def __init__(self, base_model, lga_mean=LGA_MEAN, lga_std=LGA_STD):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('lga_mean', torch.tensor(lga_mean, dtype=torch.float32))
        self.register_buffer('lga_std', torch.tensor(lga_std, dtype=torch.float32))

    def forward(self, frames):
        result = self.base_model(frames)
        y_hat_log, frame_features, context_vector, attention_scores = result
        y_hat_days = torch.exp((y_hat_log * self.lga_std) + self.lga_mean)

        return y_hat_days


class BasicAdditiveAttentionWithMasking(nn.Module):
    """
    Modified BasicAdditiveAttention that properly handles variable length sequences
    through masking in the softmax computation, but keeps the same computation
    pattern as the original BasicAdditiveAttention.
    """

    def __init__(self, input_dim: int = 1000, attention_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.linear_in = nn.Linear(in_features=input_dim, out_features=attention_dim)
        self.linear_out = nn.Linear(in_features=attention_dim, out_features=1)

    def forward(self, inputs: torch.Tensor, seq_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [batch_size, max_seq_length, input_dim] - padded input
            seq_length: [batch_size] - actual sequence length for each batch

        Returns:
            context_vector: [batch_size, input_dim]
            attention_weights: [batch_size, max_seq_length, 1]
        """
        _, max_seq_len, _ = inputs.shape

        # Compute the attention weights (same as original BasicAdditiveAttention)
        attention_weights = self.linear_in(inputs)
        attention_weights = torch.tanh(attention_weights)
        attention_weights = self.linear_out(attention_weights)

        # Create mask for sequence lengths
        range_tensor = torch.arange(max_seq_len, dtype=torch.long, device=inputs.device)
        range_tensor = range_tensor.unsqueeze(0)
        seq_length_expanded = seq_length.unsqueeze(1)
        mask = range_tensor < seq_length_expanded
        mask = mask.unsqueeze(-1)

        # Apply mask by setting padded positions to very negative value
        masked_scores = attention_weights.masked_fill(~mask, -1e9)

        # Apply softmax - now padded positions will have ~0 probability (same dim as original)
        attention_weights = torch.softmax(masked_scores, dim=1)

        # Compute the context vector (same as original BasicAdditiveAttention)
        context_vector = attention_weights * inputs  # Element-wise multiplication
        context_vector = torch.sum(context_vector, dim=1)  # Sum along sequence dimension

        return context_vector, attention_weights


class OptimizedVariableLengthModel(nn.Module):
    """
    More memory-efficient version that processes frames sequentially.
    Better for mobile deployment with limited memory.
    """

    def __init__(self, original_model, max_sequence_length=100, lga_mean=LGA_MEAN, lga_std=LGA_STD):
        super().__init__()
        self.max_sequence_length = max_sequence_length

        # Store normalization parameters as buffers (not trainable, but part of model state)
        self.register_buffer('lga_mean', torch.tensor(lga_mean, dtype=torch.float32))
        self.register_buffer('lga_std', torch.tensor(lga_std, dtype=torch.float32))

        # Extract components
        self.preprocess = original_model.cnn.preprocess
        self.cnn = original_model.cnn
        # self.attention = BasicAdditiveAttentionWithMasking(
        #     input_dim=1000,
        #     attention_dim=16
        # )
        self.attention = BasicAdditiveAttention(
            input_dim=1000,
            attention_dim=16
        )

        # Copy weights
        if hasattr(original_model, 'rnn'):
            self.attention.linear_in.load_state_dict(
                original_model.rnn.linear_in.state_dict()
            )
            self.attention.linear_out.load_state_dict(
                original_model.rnn.linear_out.state_dict()
            )

        self.regressor = original_model.regressor
    

    def extract_frame_features(self, frame: torch.Tensor):
        """frame: [N, 3, H, W] -> returns [N, feature_dim]"""
        return self.cnn(frame)

    def forward(self, frames: torch.Tensor, seq_length: torch.Tensor, feature_dim: int = 1000):
        """
        Vectorized forward: fastest for export and inference if memory allows.

        frames: [batch, T, 3, H, W]
        seq_length: [batch]
        """
        batch, T, C, H, W = frames.shape
        # Flatten frames -> process every frame in one batched cnn call
        frames_flat = frames.view(batch * T, C, H, W)        # [B*T, C, H, W]
        frames_flat = self.preprocess(frames_flat)
        features_flat = self.extract_frame_features(frames_flat)  # [B*T, D]
        features = features_flat.view(batch, T, feature_dim)      # [B, T, D]

        # Mask out padded frames
        device = frames.device
        t_range = torch.arange(T, device=device).unsqueeze(0)     # [1, T]
        mask = (t_range < seq_length.unsqueeze(1)).unsqueeze(-1)  # [B, T, 1]
        features = features * mask.float()

        context_vector, attention_weights = self.attention(features)#, seq_length)
        output_log = self.regressor(context_vector)
        # Squeeze to remove extra dimension like the original model does
        output_log = output_log.squeeze()

        # Rescale from log GA pred to days
        output_days = torch.exp((output_log * self.lga_std) + self.lga_mean)

        # Return same structure as original model with report_intermediates=True
        # (y_hat, frame_features, context, attention)
        return features, attention_weights, output_days, context_vector


def export(args):
    dicomlist = enumerate_dicom_files(args.dicom, args.examdir)

    # Raise an error if the task is not recognized
    if args.task not in ALLOWED_TASKS:
        raise ValueError(
            f"Task {args.task} not recognized. Please use one of the following: {ALLOWED_TASKS}."
        )

    print(f"Using hard-coded log_GA mean {LGA_MEAN} and std {LGA_STD}.")

    if args.task == "GA":
        return GA_TASK_EXPORT(args, dicomlist)
    if args.task == "GA_FRAMEWISE":
        return GA_TASK_EXPORT_FRAMEWISE(args, dicomlist)
    elif args.task == "FP":
        return FP_TASK(args, dicomlist, LGA_MEAN, LGA_STD)


def GA_TASK_EXPORT(args, dicomlist):
    """Exports GA model to TFLite format.

    Args:
        args (argparse.Namespace):  Parsed command-line arguments containing model
                                    and inference options.
        dicomlist (list[str]):      List of DICOM file paths to analyze.
    """
    # Load Cnn2RnnRegressor model for GA task
    base_model = load_Cnn2RnnRegressor_model(args.modelpath, args.device, args.cnn_name)

    # Use mdoel wrapper with log GA -> days rescaling
    model = ModelWithRescaling(base_model, LGA_MEAN, LGA_STD)
    model.eval()
    print(model)

    dicompath = dicomlist[0]
    frames = get_dicom_frames(dicompath, device=args.device, mode="GA")

    print(f"Exporting model to TFLite format with sequence length of {args.sequence_length}...")
    print(f"Frames shape: {frames.shape}, dtype: {frames.dtype}")

    if frames.shape[1] > args.sequence_length:
        print(f"Warning: DICOM has {frames.shape[1]} frames, which exceeds the max sequence length of {args.sequence_length}. Truncating.")
        frames = frames[:, :args.sequence_length, :, :, :]
    elif frames.shape[1] < args.sequence_length:
        print(f"Warning: DICOM has {frames.shape[1]} frames, which is less than the max sequence length of {args.sequence_length}. Padding with zeros.")
        c, h, w = frames.shape[2], frames.shape[3], frames.shape[4]
        pad_length = args.sequence_length - frames.shape[1]
        pad_tensor = torch.zeros((frames.shape[0], pad_length, c, h, w), device=args.device)
        frames = torch.cat((frames, pad_tensor), dim=1)

    edge_model = ai_edge_torch.convert(model, (frames,))

    print(edge_model)
    print("Exporting...")

    output_dir = Path("tflite_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    tflite_filename = output_dir / f"unified_model_{args.sequence_length}.tflite"
    edge_model.export(str(tflite_filename))
    print(f"TFLite model saved to {tflite_filename}")

    print(f"Exporting model with optimization: {tf.lite.Optimize.DEFAULT}...")
    quantized_model = ai_edge_torch.convert(
        model,
        (frames,),
        _ai_edge_converter_flags={"optimizations": [tf.lite.Optimize.DEFAULT]},
    )
    quantized_tflite_filename = output_dir / f"unified_model_opt_{args.sequence_length}.tflite"
    quantized_model.export(str(quantized_tflite_filename))
    print(f"Quantized TFLite model saved to {quantized_tflite_filename}")

def GA_TASK_EXPORT_FRAMEWISE(args, dicomlist):
    max_sequence_length = args.sequence_length
    original_model = load_Cnn2RnnRegressor_model(args.modelpath, args.device, args.cnn_name)
    model = OptimizedVariableLengthModel(original_model, max_sequence_length)
    model.eval()

    batch_size = 1
    sample_frames = torch.randint(0, 255, (batch_size, max_sequence_length, 3, 256, 256), dtype=torch.uint8).float()
    sample_seq_length = torch.tensor([max_sequence_length], dtype=torch.long)

    print(f"Exporting model with max sequence length {max_sequence_length}...")

    with torch.no_grad():
        edge_model = ai_edge_torch.convert(model, (sample_frames, sample_seq_length))

    output_path = f"tflite_models/variable_length_model_max{max_sequence_length}.tflite"
    edge_model.export(output_path)

    print(f"Model exported to {output_path}")
    return output_path


def FP_TASK(args, dicomlist, lmean, lstd):
    """
    Perform the Fetal Presentation (FP) task inference.
    This function loads the appropriate model, performs video-level inference,
    and optionally performs exam-level inference if multiple DICOM files are provided.

    Args:
        args (argparse.Namespace):  Parsed command-line arguments containing model
                                    and inference options.
        dicomlist (list[str]):      List of DICOM file paths to analyze.
        lmean (float):              Mean value for log normalization.
        lstd (float):               Standard deviation for log normalization.

    Returns:
        dict: Dictionary containing inference results for the FP task.
    """
    # Step 3b. Load Cnn2RnnClassifier model for FP task
    model = load_Cnn2RnnClassifier_model(
        args.modelpath,
        args.device,
        args.cnn_name,
        weights_name="IMAGENET1K_V2",
        cnn_layer_id=18,
    )

    # Step 4b. Video-level inferences for FP
    results = video_level_inference_FP(
        model,
        dicomlist,
        args.outdir,
        args.save_vectors,
        lmean=lmean,
        lstd=lstd,
    )

    # Step 5b. [Optional] Run exam-level inference for FP.
    if len(dicomlist) > 1:
        results = exam_level_inference_FP(results)

    # Step 6. Return results
    # ! Delete frame_features first
    del results["frame_features"]
    return results


def main():
    # Configure the ArgumentParser
    cli_description = (
        "Evaluate one or more DICOM files using a provided model checkpoint."
    )
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        choices=ALLOWED_TASKS,
        help="Task to perform: 'GA' (Gestational Age) or 'FP' (Fetal Presentation).",
    )
    parser.add_argument(
        "--modelpath",
        required=True,
        type=str,
        help="Path to the model checkpoint file. This should be a .ckpt file containing "
        + "the trained model weights and architecture.",
    )
    parser.add_argument(
        "--cnn_name",
        required=True,
        type=str,
        help="Name of the CNN model to use. Must be the CNN model used in model training "
        + "for the loaded checkpoint.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        type=str,
        help=f"Device to run the inference on. Default is {DEFAULT_DEVICE}. "
        + "Use 'cuda:<id>' for GPU support if available.",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        type=str,
        help="Output directory for saving prediction .csv file. Default is "
        + f"{DEFAULT_OUTDIR}. The directory will be created if it does not exist.",
    )
    parser.add_argument(
        "--save_vectors",
        action="store_true",
        help="Flag to save frame features, attention scores, and context vectors to the "
        + "output directory.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Flag to save attention scores plots to the output directory (GA task only).",
    )
    parser.add_argument(
        "--sequence_length",
        default=100,
        type=int,
        help="Max sequence length for input",
    )

    # Specify the input options for the input files or directories.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dicom",
        action="append",
        type=str,
        help="Input DICOM file(s) to the algorithm. If multiple files are specified, "
        + "they are treated as an exam. Exams are analyzed on the video and exam level.",
    )
    group.add_argument(
        "--examdir",
        default=None,
        type=str,
        help="Input directory containing DICOM file(s) to consider as an exam. Results are "
        + "tabulated on the video and exam level.",
    )

    args = parser.parse_args()
    _ = export(args)

if __name__ == "__main__":
    main()
