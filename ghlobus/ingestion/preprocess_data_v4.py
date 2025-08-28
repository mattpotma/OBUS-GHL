"""
preprocess_data_v4.py

This module preprocesses raw data from

.../ML-Raw-Data/FAMLI/GA_NEJME

Preprocessing comprises:

- crop ultrasound "fan" region
- rescale and pad to final dimensions
- standardization of target value
- write to pytorch file, spreadsheet

To avoid memory crashes, raw input data is processed by batches of ~10,000 videos.

Author: Daniel Shea
        Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd

from functools import partial
from joblib import Parallel, delayed

from ghlobus.utilities.data_utils import preprocess_video
from ghlobus.utilities.data_utils import create_video_df
from ghlobus.utilities.constants import VERY_LARGE_NUMBER


# noinspection PyUnboundLocalVariable
def main():
    # Default yaml file
    DEFAULT_YAML = "configs/GA_NEJME_train_preproc_v4.yaml"

    # Configure the ArgumentParser
    cli_description = 'Preprocess DICOM files into frame files. '
    cli_description += 'RGB frames are extracted; and they are: '
    cli_description += '- cropped to the active area; '
    cli_description += '- resized via the target `alpha`; '
    cli_description += '- padded to the final dimension; '
    cli_description += '- saved to the destination by exam folder.'

    # Add arguments to be read from the command line
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--yaml', default=DEFAULT_YAML, type=str)

    # Extract command line arguments
    args = parser.parse_args()

    # read parameter yaml file
    with open(args.yaml, 'r') as f:
        info = yaml.safe_load(f)

    # get the raw and csv file directory
    raw_dir = info['input']['raw_dir']
    csv_dir = info['input']['csv_dir']
    csv_file = info['input']['csv_file']

    # create destination directory if not there
    out_dir = info['output']['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # add root output paths to info['image'] parameter dictionary
    info['image']['out_dir'] = out_dir
    # ingestion_v4 does not save raw or jpg files
    info['image']['raw'] = None
    info['image']['jpg'] = None
    # ingestion_v4 does not have a top level folder for pt files
    info['image']['pt'] = ""
    # disable thresholds used to filter videos in preprocess_video()
    info['image']['min_frames'] = 0
    info['image']['doppler_rgb_thresh'] = VERY_LARGE_NUMBER
    info['image']['doppler_ybr_thresh'] = VERY_LARGE_NUMBER
    info['image']['doppler_pixel_thresh'] = VERY_LARGE_NUMBER

    # print information about preprocessing run
    print_string = f"Data will be written here {out_dir},\n"
    print_string += f"Prototype file is {csv_file},\n"
    print_string += f"Using {info['processing']['cores']} CPU cores,\n"
    print_string += f"Target alpha is {info['image']['alpha']},\n"
    print_string += f"Number of channels is {info['image']['channels']},\n"
    print_string += f"Frame dimensions is {(info['image']['img_size'],)*2}."
    print(print_string)

    # define partial function with arguments specified by image dictionary
    # this single argument function is needed for multi-thread processing
    preprocess_with_args = partial(preprocess_video, **info['image'])

    # Read the input file names from the yaml file
    proto_path = os.path.join(csv_dir, csv_file)
    proto_df = pd.read_csv(proto_path)
    # create absolute path (rp + '.pt' only works because the
    # base file name in relpath does not have a file extension)
    proto_df['outpath'] = [os.path.join(out_dir, rp + '.pt')
                           for rp in proto_df['relpath']]
    # Compute log_ga_boe in the subset
    proto_df['log_ga_boe'] = np.log(proto_df['ga_boe'])
    # print more information
    print(f"Created log-GA column.")

    # Create all the folders for the process:
    out_paths = pd.Series([os.path.join(out_dir, os.path.dirname(x))
                           for x in proto_df['outpath']])
    for outpath in out_paths.unique():
        os.makedirs(outpath, exist_ok=True)
    # more information output
    print(f"Completed {out_dir} subfolder creation.")

    print(f"Working on {info['output']['out_file']} ...")

    # determine number of videos in this subset
    n_videos = proto_df.shape[0]
    # partitions should be at every batch_size videos
    target_idx = list(np.arange(0, n_videos, info['processing']['batch_size'])[1:])
    # find unique exams and where they start
    _, exam_idx = np.unique(proto_df['exam_dir'].values, return_index=True)
    # match up target indices and exam boundaries
    batch_end = [exam_idx[np.argmin(np.abs(exam_idx - idx))] for idx in target_idx] + [n_videos]
    n_batches = len(batch_end)
    batch_dfs = [None for _ in range(n_batches)]

    # iterate over partitions
    beg = 0
    for batch_idx, end in enumerate(batch_end):
        # grab the dataframe corresponding to this partition
        batch_proto_df = proto_df.iloc[beg:end].copy()
        batch_proto_df.reset_index(inplace=True, drop=True)
        # reset beginning of next partition to where we are ending this partition
        beg = end

        # grab paths for this subset
        relpaths = batch_proto_df['relpath'].tolist()
        file_types = batch_proto_df['file_type'].tolist()
        in_filepaths = [os.path.join(raw_dir, x) for x in relpaths]
        projects = [x.split(os.path.sep)[0] for x in relpaths]
        exam_dirs = [x.split(os.path.sep)[1] for x in relpaths]
        tags = batch_proto_df['tag'].tolist()
        # stack file-specific information into DataFrame for passing to worker
        file_info_df = pd.DataFrame({
            'in_filepath': in_filepaths,
            'file_type': file_types,
            'project': projects,
            'exam_dir': exam_dirs,
            'tag': tags,
            'pdx': [None] * len(in_filepaths)})

        # Launch job; get list of results
        if info['processing']['cores'] > 1:
            # Configure parallelization
            par_job = Parallel(n_jobs=info['processing']['cores'], verbose=True)

            # process this batch in parallel to a list of dataframes
            batch_shape_df_lst = par_job(delayed(preprocess_with_args)(f)
                                         for _, f in file_info_df.iterrows())

            # clear parallelization to conserve memory
            del par_job

            # Stack this batch's output into a dataframe to be stacked with other batches
            batch_shape_df = pd.concat(batch_shape_df_lst, axis=0)
        else:
            # non-parallelized for debugging
            dummy_df = create_video_df(
                reason="dummy",
                shape_in=(0, 0, 0, 0),
                shape_out=(0, 0, 0, 0),
                bbox=(0, 0, 0, 0),
            )
            batch_shape_df = pd.DataFrame(columns=dummy_df.columns)

            # process this batch one file at a time
            for _, f in file_info_df.iterrows():
                video_shape_df = preprocess_with_args(f)
                batch_shape_df = pd.concat((batch_shape_df, video_shape_df), axis=0)

        # concatenate columns from prototype DF and shape DF
        batch_shape_df.reset_index(inplace=True, drop=True)
        batch_df = pd.concat((batch_proto_df, batch_shape_df), axis=1)

        # append this batch's dataframe to the list of batch dataframes
        batch_dfs[batch_idx] = batch_df

    # stack the batch_dfs list into a single dataframe
    output_df = pd.concat(batch_dfs, axis=0)
    # sort by 'exam_dir` column
    output_df.sort_values(by=['relpath'], inplace=True)
    # reset the index
    output_df.reset_index(drop=True, inplace=True)

    # Save the output_df
    csv_file_path = os.path.join(out_dir, info['output']['out_file'])
    output_df.to_csv(csv_file_path, index=False)


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
