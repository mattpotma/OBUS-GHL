"""
tflite_inference.py

A script for executing TFLite inference pipeline on one or more DICOM files.
Based on the original inference.py but adapted for TensorFlow Lite models.

This script mimics the original inference.py but uses TFLite models instead of PyTorch.
Currently only supports the GA (Gestational Age) task.
"""

import argparse
import os
import numpy as np
import time

import torch
import tensorflow as tf

from ghlobus.utilities.inference_utils import (
    enumerate_dicom_files,
    create_output_directories,
    write_results,
    get_dicom_frames,
    LGA_MEAN,
    LGA_STD,
    resample_frames,
)
from ghlobus.utilities.sample_utils import matern_subsample

# Set defaults
DEFAULT_TFLITE_DIR = "./tflite_models/"
DEFAULT_OUTDIR = "./test_output_tflite/"
ALLOWED_TASKS = ["GA", "FP", "EFW", "TWIN"]
PRESENTATION_LABELS = {0: "cephalic", 1: "noncephalic"}
DEFAULT_TWIN_BAG_SIZE = 1000


def load_tflite_model(tflite_path: str):
    """
    Load a TensorFlow Lite model.

    Args:
        tflite_path (str): Path to the TFLite model file.

    Returns:
        tf.lite.Interpreter: The loaded TFLite interpreter.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def tflite_inference(interpreter, input_data, seq_length=None):
    """
    Run inference using TensorFlow Lite interpreter.

    Args:
        interpreter: TFLite interpreter
        input_data: Input numpy array (float32)
        seq_length: Optional sequence length for variable length models

    Returns:
        numpy array: Model output (prediction)
    """
    start = time.time()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")

    # Get input shape
    input_shape = input_details[0]["shape"]
    print(f"Model expects input shape: {input_shape}")

    # Convert input to the expected type (UINT8 as expected by the model)
    if input_details[0]["dtype"] == np.uint8:
        # The input is already in the correct range (0-254), just convert dtype
        input_quantized = input_data.astype(np.uint8)
    else:
        input_quantized = input_data.astype(input_details[0]["dtype"])

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], input_quantized)

    # Set sequence length tensor if this is a variable length model
    if len(input_details) > 1 and seq_length is not None:
        seq_length_tensor = np.array([seq_length], dtype=np.int64)
        interpreter.set_tensor(input_details[1]["index"], seq_length_tensor)
        print(f"Set sequence length: {seq_length}")

    # Run inference
    interpreter.invoke()

    output_data = [
        interpreter.get_tensor(output_details[i]["index"])
        for i, _ in enumerate(output_details)
    ]

    if len(output_data) == 1:
        output_data = output_data[0]

    end = time.time()
    print(f"TFLite inference time: {end - start:.4f} seconds")

    return output_data


def video_level_inference_tflite(
    interpreter, dicomlist, outdir, lmean=LGA_MEAN, lstd=LGA_STD, sequence_length=None
):
    """
    Run TFLite inference on each DICOM file individually.

    Args:
        interpreter: TFLite interpreter
        dicomlist: List of DICOM file paths
        outdir: Output directory
        lmean: Mean for log normalization
        lstd: Standard deviation for log normalization

    Returns:
        dict: Results dictionary
    """
    results = {
        "paths": [],
        "Predicted Gestational Age (Days)": [],
    }

    for dicompath in dicomlist:
        print(f"Processing: {dicompath}")

        # Step 1: Prepare frames from DICOM
        # Note: get_dicom_frames returns torch.Tensor (uint8), we need to convert to numpy
        frames_torch = get_dicom_frames(dicompath, device="cpu", mode="GA")
        # Input shape: (1, 110, 3, 256, 256)

        frames_torch = resample_frames(
            frames_torch,
            sequence_length=sequence_length,
            channels=frames_torch.shape[2],
            hw=(frames_torch.shape[3], frames_torch.shape[4]),
        )

        frames_numpy = (
            frames_torch.detach().cpu().numpy()
        )  # Keep original dtype (uint8)

        # Step 2: Run TFLite inference
        y_hat_days = tflite_inference(interpreter, frames_numpy, frames_numpy.shape[0])

        # Step 3: Extract scalar values (handle different output shapes)
        if y_hat_days.size > 1:
            y_hat_days_scalar = float(np.mean(y_hat_days))
        else:
            y_hat_days_scalar = float(y_hat_days.item())

        # Step 4: Print results
        print(f"Predicted GA (Days): {y_hat_days_scalar:.2f}")

        # Step 5: Store results
        results["paths"].append(dicompath)
        results["Predicted Gestational Age (Days)"].append(y_hat_days_scalar)

    return results


def video_level_inference_tflite_fp(
    interpreter, dicomlist, outdir, sequence_length=None
):
    """
    Run TFLite FP inference on each DICOM file individually.
    """
    results = {
        "paths": [],
        "Predicted presentation": [],
        "Probabilities": [],
    }

    for dicompath in dicomlist:
        print(f"Processing: {dicompath}")

        frames_torch = get_dicom_frames(dicompath, device="cpu", mode="FP")

        frames_torch = resample_frames(
            frames_torch,
            sequence_length=sequence_length,
            channels=frames_torch.shape[2],
            hw=(frames_torch.shape[3], frames_torch.shape[4]),
        )

        frames_numpy = frames_torch.detach().cpu().numpy()

        probs = tflite_inference(interpreter, frames_numpy, frames_numpy.shape[0])[1]
        probs = np.squeeze(probs)

        print(probs)
        pred_idx = int(np.argmax(probs))
        pred_label = PRESENTATION_LABELS.get(pred_idx, str(pred_idx))

        print(f"Predicted presentation: {pred_label} (prob={probs[pred_idx]:.4f})")

        results["paths"].append(dicompath)
        results["Predicted presentation"].append(pred_label)
        results["Probabilities"].append(probs.tolist())

    return results


def video_level_inference_tflite_efw(
    interpreter, dicomlist, outdir, sequence_length=None
):
    """
    Run TFLite EFW inference on each DICOM file individually.
    """
    results = {
        "paths": [],
        "Predicted AC (mm)": [],
        "Predicted FL (mm)": [],
        "Predicted HC (mm)": [],
        "Predicted BPD (mm)": [],
        "Predicted EFW (g)": [],
    }

    for dicompath in dicomlist:
        print(f"Processing: {dicompath}")

        frames_torch = get_dicom_frames(dicompath, device="cpu", mode="EFW")
        original_seq_length = frames_torch.shape[1]

        if sequence_length is not None and frames_torch.shape[1] > sequence_length:
            indices = np.linspace(0, frames_torch.shape[1] - 1, sequence_length).astype(
                int
            )
            frames_torch = frames_torch[:, indices, :, :, :]
            original_seq_length = sequence_length

        if sequence_length is not None and frames_torch.shape[1] < sequence_length:
            pad_length = sequence_length - frames_torch.shape[1]
            pad_tensor = torch.zeros(
                (1, pad_length, 3, 256, 256), dtype=frames_torch.dtype
            )
            frames_torch = torch.cat((frames_torch, pad_tensor), dim=1)

        frames_numpy = frames_torch.detach().cpu().numpy()

        print(f"Input shape: {frames_numpy.shape}")
        print(f"Original sequence length: {original_seq_length}")

        outputs = tflite_inference(interpreter, frames_numpy, original_seq_length)
        outputs = np.squeeze(outputs)

        if outputs.ndim == 0:
            outputs = np.expand_dims(outputs, 0)

        if outputs.size < 5:
            raise ValueError(
                "EFW TFLite model must output five values (AC, FL, HC, BPD, EFW)."
            )

        ac_pred, fl_pred, hc_pred, bpd_pred, efw_pred = outputs[:5]

        print(
            f"Predicted AC: {ac_pred:.6f} mm, FL: {fl_pred:.6f} mm, HC: {hc_pred:.6f} mm, "
            f"BPD: {bpd_pred:.6f} mm, EFW: {efw_pred:.6f} g"
        )

        results["paths"].append(dicompath)
        results["Predicted AC (mm)"].append(float(ac_pred))
        results["Predicted FL (mm)"].append(float(fl_pred))
        results["Predicted HC (mm)"].append(float(hc_pred))
        results["Predicted BPD (mm)"].append(float(bpd_pred))
        results["Predicted EFW (g)"].append(float(efw_pred))

    return results


def exam_level_inference_tflite_twin(
    interpreter, dicomlist, outdir, bag_size=None
):
    """
    Run TFLite TWIN inference for a 6-video exam and return probs + pred.
    """
    if len(dicomlist) != 6:
        raise ValueError(
            "For TWIN TFLite inference, exactly 6 DICOM files must be provided. "
            f"Found {len(dicomlist)} files."
        )

    exam_frames = []
    for dicompath in dicomlist:
        frames = get_dicom_frames(dicompath, device="cpu")
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames.squeeze(0)
        exam_frames.append(frames)

    all_frames = torch.cat(exam_frames, dim=0)
    total_frames = all_frames.shape[0]

    if bag_size is None:
        bag_size = DEFAULT_TWIN_BAG_SIZE

    if total_frames >= bag_size:
        indices = matern_subsample(total_frames, k=bag_size)
        sampled_frames = all_frames[indices]
    else:
        pad_length = bag_size - total_frames
        c, h, w = all_frames.shape[1], all_frames.shape[2], all_frames.shape[3]
        pad_tensor = torch.zeros(
            (pad_length, c, h, w),
            dtype=all_frames.dtype,
        )
        sampled_frames = torch.cat((all_frames, pad_tensor), dim=0)

    sampled_frames = sampled_frames.unsqueeze(0)  # (1, bag_size, C, H, W)
    frames_numpy = sampled_frames.detach().cpu().numpy()

    outputs = tflite_inference(interpreter, frames_numpy)
    if not isinstance(outputs, (list, tuple)) or len(outputs) == 0:
        raise ValueError("TWIN TFLite model did not return any outputs.")

    # Prefer the tensor that looks like probabilities (last dim == 2)
    probs = None
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim >= 1 and arr.shape[-1] == 2:
            probs = arr
            break

    if probs is None:
        raise ValueError(
            "TWIN TFLite model must output a probabilities tensor with last dimension 2."
        )

    probs = np.squeeze(probs)
    pred = int(np.argmax(probs))
    pred_label = "Twin" if pred == 1 else "Singleton"

    print(probs)
    print(f"Predicted label: {pred_label} (prob={probs[pred]:.4f})")

    results = {
        "paths": [os.path.dirname(dicomlist[0])],
        "Predicted label": [pred],
        "Probabilities": [probs.tolist()],
    }

    return results


def GA_TASK_tflite(args, dicomlist, lmean, lstd):
    """
    Perform the Gestational Age (GA) task inference using TFLite.

    Args:
        args: Parsed command-line arguments
        dicomlist: List of DICOM file paths
        lmean: Mean value for log normalization
        lstd: Standard deviation for log normalization

    Returns:
        dict: Dictionary containing inference results
    """
    # Load TFLite model
    print("\n\n" + "=" * 80)
    print(
        f"Loading TFLite model from: {args.tflite_dir} and model name {args.tflite_model_name}"
    )
    tflite_model_path = os.path.join(args.tflite_dir, args.tflite_model_name)
    if not os.path.exists(tflite_model_path):
        raise FileNotFoundError(f"TFLite model not found at: {tflite_model_path}")

    interpreter = load_tflite_model(tflite_model_path)

    # Run video-level inference
    results = video_level_inference_tflite(
        interpreter,
        dicomlist,
        args.outdir,
        lmean=lmean,
        lstd=lstd,
        sequence_length=args.sequence_length,
    )

    return results


def FP_TASK_tflite(args, dicomlist):
    """
    Perform the FP task inference using TFLite.
    """
    print("\n\n" + "=" * 80)
    print(
        f"Loading FP TFLite model from: {args.tflite_dir} and model name {args.tflite_model_name}"
    )
    tflite_model_path = os.path.join(args.tflite_dir, args.tflite_model_name)
    if not os.path.exists(tflite_model_path):
        raise FileNotFoundError(f"TFLite model not found at: {tflite_model_path}")

    interpreter = load_tflite_model(tflite_model_path)

    results = video_level_inference_tflite_fp(
        interpreter, dicomlist, args.outdir, sequence_length=args.sequence_length
    )

    return results


def EFW_TASK_tflite(args, dicomlist):
    """
    Perform the EFW task inference using TFLite.
    """
    print("\n\n" + "=" * 80)
    print(
        f"Loading EFW TFLite model from: {args.tflite_dir} and model name {args.tflite_model_name}"
    )
    tflite_model_path = os.path.join(args.tflite_dir, args.tflite_model_name)
    if not os.path.exists(tflite_model_path):
        raise FileNotFoundError(f"TFLite model not found at: {tflite_model_path}")

    interpreter = load_tflite_model(tflite_model_path)

    results = video_level_inference_tflite_efw(
        interpreter, dicomlist, args.outdir, sequence_length=args.sequence_length
    )

    return results


def TWIN_TASK_tflite(args, dicomlist):
    """
    Perform the TWIN task inference using TFLite.
    """
    print("\n\n" + "=" * 80)
    print(
        f"Loading TWIN TFLite model from: {args.tflite_dir} and model name {args.tflite_model_name}"
    )
    tflite_model_path = os.path.join(args.tflite_dir, args.tflite_model_name)
    if not os.path.exists(tflite_model_path):
        raise FileNotFoundError(f"TFLite model not found at: {tflite_model_path}")

    interpreter = load_tflite_model(tflite_model_path)

    results = exam_level_inference_tflite_twin(
        interpreter,
        dicomlist,
        args.outdir,
        bag_size=args.sequence_length,
    )

    return results


def inference(args):
    """
    Main inference function.

    Args:
        args: Parsed command-line arguments

    Returns:
        dict: Inference results
    """
    # Step 1: Create the dicomlist
    dicomlist = enumerate_dicom_files(args.dicom, args.examdir)

    # Step 2: Create output directories
    create_output_directories(
        args.outdir, False, False
    )  # No vector/plot saving for TFLite

    # Step 3: Validate task
    if args.task not in ALLOWED_TASKS:
        raise ValueError(
            f"Task {args.task} not recognized. "
            f"Please use one of the following: {ALLOWED_TASKS}."
        )

    # Step 4: Get normalization constants
    print(f"Using hard-coded log_GA mean {LGA_MEAN} and std {LGA_STD}.")

    # Step 5: Perform task-specific inference
    if args.task == "GA":
        return GA_TASK_tflite(args, dicomlist, LGA_MEAN, LGA_STD)
    elif args.task == "FP":
        return FP_TASK_tflite(args, dicomlist)
    elif args.task == "EFW":
        return EFW_TASK_tflite(args, dicomlist)
    elif args.task == "TWIN":
        return TWIN_TASK_tflite(args, dicomlist)
    else:
        raise ValueError(f"Task {args.task} not supported in TFLite inference yet.")


def main():
    """Main function for command-line usage."""
    # Configure the ArgumentParser (mirror original)
    cli_description = "Evaluate one or more DICOM files using TensorFlow Lite models."
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        choices=ALLOWED_TASKS,
        help="Task to perform: 'GA' (Gestational Age), 'FP' (Fetal Presentation), "
        "'EFW' (Estimated Fetal Weight), or 'TWIN' (Multiple Gestation)",
    )
    parser.add_argument(
        "--tflite_dir",
        default=DEFAULT_TFLITE_DIR,
        type=str,
        help=f"Directory containing TFLite model files. Default is {DEFAULT_TFLITE_DIR}.",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        type=str,
        help="Output directory for saving prediction .csv file. Default is "
        + f"{DEFAULT_OUTDIR}. The directory will be created if it does not exist.",
    )
    parser.add_argument(
        "--tflite_model_name",
        default="unified_model.tflite",
        type=str,
        help="Name of the TFLite model file to use. Default is 'unified_model.tflite'.",
    )
    parser.add_argument(
        "--sequence_length",
        default=None,
        type=int,
        help="Optional sequence length to pad/truncate input frames to. Default is None (use all frames).",
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

    # Extract command line arguments
    args = parser.parse_args()

    # Run TFLite inference
    results = inference(args)

    # Write the results to the outdir
    write_results(results, args.outdir)


if __name__ == "__main__":
    main()
