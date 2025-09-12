"""
inference.py

A script for executing inference pipeline on one or more DICOM files.

Author: Dan Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""

import argparse
import tensorflow as tf

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

# Set default_model and default_dicom
DEFAULT_DEVICE = "cpu"
DEFAULT_OUTDIR = "./test_output/"
ALLOWED_TASKS = ["GA", "FP"]


def inference(args):
    # Step 1. Create the `dicomlist`, which is a list of
    # filepaths to DICOM files for analysis
    dicomlist = enumerate_dicom_files(args.dicom, args.examdir)

    # Step 2. Create the output directories
    create_output_directories(
        args.outdir, args.save_vectors, args.task == "GA" and args.save_plots
    )

    # Raise an error if the task is not recognized
    if args.task not in ALLOWED_TASKS:
        raise ValueError(
            f"Task {args.task} not recognized. Please use one of the following: {ALLOWED_TASKS}."
        )

    # Step 3. Get the Z-scale log normalization constants
    print(f"Using hard-coded log_GA mean {LGA_MEAN} and std {LGA_STD}.")

    # Step 4. Perform the task-specific inference
    # noinspection PyInconsistentReturns
    if args.task == "GA":
        return GA_TASK_EXPORT(args, dicomlist)
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
    model = load_Cnn2RnnRegressor_model(args.modelpath, args.device, args.cnn_name)
    print(model)

    dicompath = dicomlist[0]
    frames = get_dicom_frames(dicompath, device=args.device, mode="GA")

    print("Exporting model to TFLite format...")
    print(f"Frames shape: {frames.shape}, dtype: {frames.dtype}")

    edge_model = ai_edge_torch.convert(model.eval(), (frames,))

    print(edge_model)
    print("Exporting...")
    tflite_filename = "tflite_models/unified_model.tflite"
    edge_model.export(tflite_filename)
    print(f"TFLite model saved to {tflite_filename}")

    print(f"Exporting model with optimization: {tf.lite.Optimize.DEFAULT}...")
    quantized_model = ai_edge_torch.convert(
        model.eval(),
        (frames,),
        _ai_edge_converter_flags={"optimizations": [tf.lite.Optimize.DEFAULT]},
    )
    quantized_tflite_filename = "tflite_models/unified_model_optimized_default.tflite"
    quantized_model.export(quantized_tflite_filename)
    print(f"TFLite model saved to {tflite_filename}")
    print(f"Quantized TFLite model saved to {quantized_tflite_filename}")


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
        choices=["GA", "FP"],
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

    # Run inference
    results = inference(args)

    # Write  the results to the outdir
    write_results(results, args.outdir)


if __name__ == "__main__":
    main()
