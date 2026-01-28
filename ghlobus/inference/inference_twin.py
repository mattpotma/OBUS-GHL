"""
inference.py

A script for executing inference pipeline on one or more DICOM files.

Author: Dan Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import torch
import glob
import yaml
import sys
import re
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from ghlobus.utilities.sample_utils import matern_subsample


from ghlobus.utilities.inference_utils import (
    detach_and_convert_tensors,
    get_dicom_frames,
    load_TWIN_model,
    enumerate_media_files,
    create_output_directories,
    write_results,
)

# Set default_model and default_dicom
DEFAULT_DEVICE = 'cpu'
DEFAULT_OUTDIR = './test_output/'


def inference(args):
    # Step 1. Create the `dicomlist`, which is a list of
    # filepaths to DICOM files for analysis
    print("Enumerating media files...")
    filelist = enumerate_media_files(args.file, args.examdir)

    # For Twins, must have exactly n=6 files
    if len(filelist) != 6:
        raise ValueError(f"For TWIN analysis, exactly 6 files must be provided. "
                         f"Found {len(filelist)} files.")

    # If there are `.mp4` files, ensure that pdx is provided
    if any([fpath.lower().endswith('.mp4') for fpath in filelist]):
        if args.pdx is None:
            raise ValueError("When using MP4 files, the --pdx argument "
                             "must be provided with the physical delta X "
                             "value in mm/pixel.")

    # Step 2. Create the output directories
    print("Creating output directories...")
    create_output_directories(args.outdir,
                              False,  # for creating a vectors subdir
                              False)  # for creating a plots subdir)

    # Step 3. Load Cnn2RnnRegressor model for GA task
    print("Loading model...")
    model = load_TWIN_model(
        args.modelpath, args.device, args.cnn_name)

    # Step 4. Generate the frames as input for the model.
    print("Extracting frames from media files...")
    exam_frames = []
    for i, fpath in enumerate(filelist):
        # Step 4a. Prepare the frame data from the DICOM for the model
        frames = get_dicom_frames(fpath, device=args.device, pdx=args.pdx)

        # If frames is returned as a `None`, raise an exception
        if frames is None:
            raise ValueError(f"No frames were extracted from file: {fpath}. "
                             f"Check that the file is a valid DICOM or MP4 file.")

        # Remove the leading singleton dimension if present
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames.squeeze(0)

        exam_frames.append(frames)

        # Print the shape of each video's frames
        print(f"Video {i+1} shape: {frames.shape}")

    # Step 5. Subsample the frames to `bag_size` according to `frame_sampling`
    print("Subsampling frames...")
    bag_size = args.bag_size

    def pad_to_bag_size(frames: torch.Tensor, target_size: int) -> torch.Tensor:
        """Pad frames to target_size with zeros along the frame dimension."""
        if frames.shape[0] >= target_size:
            return frames
        pad_length = target_size - frames.shape[0]
        pad_tensor = torch.zeros(
            (pad_length, *frames.shape[1:]),
            dtype=frames.dtype,
            device=frames.device,
        )
        return torch.cat((frames, pad_tensor), dim=0)

    if args.frame_sampling == 'exam':
        print("Using exam-level frame sampling.")
        print("Concatenating all frames from all videos...")
        # Concatenate all frames from all videos
        all_frames = torch.cat(exam_frames, dim=0)
        total_frames = all_frames.shape[0]

        # print shape after concatenation
        print(
            f"Total frames before sampling: {total_frames} (exam shape {all_frames.shape})")

        if total_frames <= bag_size:
            # If total frames are less than or equal to bag size, pad to bag size
            sampled_frames = pad_to_bag_size(all_frames, bag_size)
        else:
            # use Matern sampling to get separation between samples on the exam level
            indices = matern_subsample(all_frames.shape[0], k=args.bag_size)
            sampled_frames = all_frames[indices]

        # Reshape to have a batch dimension of 1
        sampled_frames = sampled_frames.unsqueeze(0)
    elif args.frame_sampling == 'video':
        # Process each video's frames independently
        sampled_videos = []
        for frames in exam_frames:
            total_frames = frames.shape[0]

            if total_frames <= bag_size:
                # If total frames are less than or equal to bag size, pad to bag size
                sampled = pad_to_bag_size(frames, bag_size)
            else:
                # use Matern sampling to get separation between samples on the video level
                indices = matern_subsample(frames.shape[0], k=args.bag_size)
                sampled = frames[indices]

            sampled_videos.append(sampled)

        # Stack the sampled videos and add a batch dimension of 1
        sampled_frames = torch.stack(sampled_videos, dim=0).unsqueeze(0)
    else:
        raise ValueError(
            f"Invalid frame_sampling argument: {args.frame_sampling}")

    print(f"Total frames after sampling: {sampled_frames.shape[1]}")
    print(f"Shape of input tensor to model: {sampled_frames.shape}")

    # Run inference
    print("Running inference...")
    meanings = {0: 'Singleton', 1: 'Twin'}
    result = model(sampled_frames)
    print("Inference complete.")
    # Unpack the result
    y_hat, _, _, _, logits = result
    # Detach and convert tensors to Numpy values.
    y_hat, logits = detach_and_convert_tensors(model, [y_hat, logits])
    y_hat = y_hat[0]
    logits = logits[0]

    # print(f"Logits:\n{logits}")
    # print(f"Prediction Scores (log-softmax):\n{np.exp(y_hat)}")
    print(f"Class {meanings[0]} score: {np.exp(y_hat[0]):<.4f}")
    print(f"Class {meanings[1]} score: {np.exp(y_hat[1]):<.4f}")
    print("------------------")

    # Compute the prediction
    # Determine value for prediction
    if args.video_prediction_threshold is not None:
        # Threshold the video-level prediction
        # Must exponentiate bcs output is log-softmax!
        prediction = int(np.exp(y_hat)[1] > args.video_prediction_threshold)
    else:
        # Use the argmax of the softmax output as the prediction as default
        prediction = np.argmax(y_hat).item()

    print(f"Predicted Class: {prediction} ({meanings[prediction]})")
    print("------------------")

    # Create the results dictionary for writing to a .csv file
    results = {
        "model": ["TWIN"],
        'exam_dir': [os.path.dirname(filelist[0])],
        'Predicted label': [prediction],
        'Predicted meaning': [meanings[prediction]],
        'SOFTMAX_NEG': [np.exp(y_hat)[0]],
        'SOFTMAX_POS': [np.exp(y_hat)[1]],
        'Logits': [logits.tolist()],
        'TotalInstances': [len(filelist)],
        "total_frames_analyzed": [int(sampled_frames.shape[1])],
    }

    return results


def main():
    # Configure the ArgumentParser
    cli_description = 'Evaluate an exam of 6 blind sweep video files using a provided model checkpoint.'
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--modelpath', required=True, type=str,
                        help="Path to the model checkpoint file. This should be a .ckpt file containing " +
                             "the trained model weights and architecture.")
    parser.add_argument('--cnn_name', required=False, type=str, default='MobileNet_V2',
                        help="Name of the CNN model to use. Must be the CNN model used in model training " +
                             "for the loaded checkpoint.")
    parser.add_argument('--device', default=DEFAULT_DEVICE, type=str,
                        help=f"Device to run the inference on. Default is {DEFAULT_DEVICE}. " +
                             "Use 'cuda:<id>' for GPU support if available.")
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR, type=str,
                        help="Output directory for saving prediction .csv file. Default is " +
                             f"{DEFAULT_OUTDIR}. The directory will be created if it does not exist.")
    parser.add_argument('--pdx', default=None, type=float,
                        help="Physical Delta X value in mm/pixel. Required when using MP4 files.")
    parser.add_argument('--frame_sampling', default='exam', type=str, choices=['video', 'exam'],
                        help="Frame sampling strategy for exam-level inference. " +
                             "This controls when frames are sampled for inference. " +
                        "'video' samples frames from each video independently, " +
                        "'exam' samples frames from all videos concatenated as one. " +
                        "Default is 'exam' to match training protocols.")
    parser.add_argument('--bag_size', default=1000, type=int,
                        help="Number of frames to sample for inference. Default is 300")
    parser.add_argument('--video_prediction_threshold', default=None, type=float,
                        help="Threshold for classifying the video-level prediction. " +
                             "If not provided, the class with the highest probability is used. " +
                             "If provided, the predicted probability of the positive class " +
                             "must exceed this threshold to be classified as positive.")
    # Specify the input options for the input files or directories.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', action='append', type=str,
                       help="Input DICOM or MP4 file(s) to the algorithm. If multiple files are specified, " +
                            "they are treated as an exam. Exams are analyzed on the video and exam level.")
    group.add_argument('--examdir', default=None, type=str,
                       help="Input directory containing DICOM file(s) to consider as an exam. Results are " +
                            "tabulated on the exam level.")

    # Extract command line arguments
    args = parser.parse_args()

    # Run inference
    results = inference(args)

    # Write  the results to the outdir
    write_results(results, args.outdir)


if __name__ == '__main__':
    main()
