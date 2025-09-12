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

import tensorflow as tf

from ghlobus.utilities.inference_utils import (
    enumerate_dicom_files,
    create_output_directories,
    write_results,
    get_dicom_frames,
    get_rescale_log_value_func,
    LGA_MEAN,
    LGA_STD,
)

# Set defaults
DEFAULT_TFLITE_DIR = "./tflite_models/"
DEFAULT_OUTDIR = "./test_output_tflite/"
ALLOWED_TASKS = ["GA"]  # Only GA supported for now


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


def tflite_inference(interpreter, input_data):
    """
    Run inference using TensorFlow Lite interpreter.

    Args:
        interpreter: TFLite interpreter
        input_data: Input numpy array (float32)

    Returns:
        numpy array: Model output (prediction)
    """
    start = time.time()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Convert input to the expected type (UINT8 as expected by the model)
    if input_details[0]["dtype"] == np.uint8:
        # The input is already in the correct range (0-254), just convert dtype
        input_quantized = input_data.astype(np.uint8)
    else:
        input_quantized = input_data.astype(input_details[0]["dtype"])

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], input_quantized)

    # Run inference
    interpreter.invoke()

    # Get output tensor - use output index 2 which is the scalar prediction
    output_data = interpreter.get_tensor(output_details[2]["index"])

    end = time.time()
    print(f"TFLite inference time: {end - start:.4f} seconds")

    return output_data


def video_level_inference_tflite(
    interpreter, dicomlist, outdir, lmean=LGA_MEAN, lstd=LGA_STD
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
        "Predicted Log Gestational Age": [],
        "Predicted Gestational Age (Days)": [],
    }

    # Create rescale function
    rescale_log_ga = get_rescale_log_value_func(lmean, lstd)

    for dicompath in dicomlist:
        print(f"Processing: {dicompath}")

        # Step 1: Prepare frames from DICOM
        # Note: get_dicom_frames returns torch.Tensor (uint8), we need to convert to numpy
        frames_torch = get_dicom_frames(dicompath, device="cpu", mode="GA")
        frames_numpy = (
            frames_torch.detach().cpu().numpy()
        )  # Keep original dtype (uint8)

        print(f"Input shape: {frames_numpy.shape}")

        # Step 2: Run TFLite inference
        y_hat_log = tflite_inference(interpreter, frames_numpy)

        # Step 3: Convert log prediction to days
        y_hat_days = rescale_log_ga(y_hat_log)

        # Step 4: Extract scalar values (handle different output shapes)
        if y_hat_log.size > 1:
            y_hat_log_scalar = float(np.mean(y_hat_log))
            y_hat_days_scalar = float(np.mean(y_hat_days))
        else:
            y_hat_log_scalar = float(y_hat_log.item())
            y_hat_days_scalar = float(y_hat_days.item())

        # Step 5: Print results
        print(f"Predicted Log GA: {y_hat_log_scalar:.4f}")
        print(f"Predicted GA (Days): {y_hat_days_scalar:.2f}")

        # Step 6: Store results
        results["paths"].append(dicompath)
        results["Predicted Log Gestational Age"].append(y_hat_log_scalar)
        results["Predicted Gestational Age (Days)"].append(y_hat_days_scalar)

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
    print(f"Loading TFLite model from: {args.tflite_dir} and model name {args.tflite_model_name}")
    tflite_model_path = os.path.join(args.tflite_dir, args.tflite_model_name)
    if not os.path.exists(tflite_model_path):
        raise FileNotFoundError(f"TFLite model not found at: {tflite_model_path}")

    interpreter = load_tflite_model(tflite_model_path)

    # Run video-level inference
    results = video_level_inference_tflite(
        interpreter, dicomlist, args.outdir, lmean=lmean, lstd=lstd
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
        choices=["GA"],
        help="Task to perform: 'GA' (Gestational Age). Only GA supported currently.",
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
