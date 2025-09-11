"""
efw_data_split_v9.py

This module splits data into train, validation, and test sets
according to specified fractions of the total data.

Author: Courosh Mehanian
        Wenlong Shi

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import sys
import yaml
import random
import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from tqdm import tqdm
from typing import Dict
from functools import partial
from collections import defaultdict

from ghlobus.utilities.constants import META_DIR
from ghlobus.utilities.plot_utils import Logger
from ghlobus.utilities.data_utils import construct_outpath
from ghlobus.utilities.data_utils import read_spreadsheet_columns
from ghlobus.utilities.sample_utils import evenly_spaced_elements
from ghlobus.utilities.sweep_utils import get_tag_selection


def split_by_patient(df: pd.DataFrame,
                     train_fraction: float = 0.75,
                     val_fraction: float = 0.1,
                     ) -> Dict[str, pd.DataFrame]:
    """
    Split the data into train, val, and test sets by patientID.
    When splitting the data, exams with the same PID should be in the same set.
    train_fraction and val_fraction are specified, and test_fraction is
    the remainder of the data.

    Args:
        df (pd.DataFrame):        Instance table containing the data.
        train_fraction (float):   Fraction of data to use for training.
        val_fraction (float):     Fraction of data to use for validation.

    Returns:
        Dict[str, pd.DataFrame]: The split study info.
    """
    # Get the patient IDs
    unique_pids = df['PID'].unique()

    # Create a dictionary to store the largest EFW value for each patient ID
    pid_largest_efw = dict()

    # Iterate through each patient ID and its associated exams
    for pid in tqdm(unique_pids,
                    desc="Finding largest EFW Hadlock values",
                    unit="PID",
                    total=len(unique_pids)):
        # Get the EFW hadlock values for this patient ID
        mask = df['PID'] == pid
        EFW_hadlock_values = df.loc[mask, 'EFW_hadlock'].values
        if len(EFW_hadlock_values) > 0:
            pid_largest_efw[pid] = max(EFW_hadlock_values)
        else:
            # If no EFW values are available, raise error
            raise ValueError(f"No EFW_hadlock found for PID: {pid}")

    # Get PIDs and EFW values for sorting in next step
    pids = np.array(list(pid_largest_efw.keys()))
    efws = np.array([float(x) for x in pid_largest_efw.values()])
    # Sort order for the patients based on largest EFW
    inds = np.argsort(efws)

    # Calculate the total number of patients
    total = len(pids)
    # Allocate train_fraction of patients to training set
    train_cnt = int(total * train_fraction)
    # Allocate val_fraction of patients to validation set
    val_cnt = int(total * val_fraction)
    # Allocate remaining patients to test set
    test_cnt = total - train_cnt - val_cnt

    # Select testing set patients using evenly spaced sampling
    inds_test = evenly_spaced_elements(inds, test_cnt)
    # Select validation set patients from remaining patients
    inds_val = evenly_spaced_elements([x for x in inds if x not in inds_test], val_cnt)
    # Assign all remaining patients to the training set
    inds_train = [x for x in inds if x not in inds_test and x not in inds_val]

    # Extract just the patient IDs for the test, val, and train sets
    pids_test = pids[inds_test]
    pids_val = pids[inds_val]
    pids_train = pids[inds_train]

    # Separate the df into train, val, and test sets based on selected PIDs
    df_train = df[df['PID'].isin(pids_train)].copy()
    df_val = df[df['PID'].isin(pids_val)].copy()
    df_test = df[df['PID'].isin(pids_test)].copy()

    # Organize into dictionary of DataFrames
    set_dfs = {
        'train': df_train,
        'val': df_val,
        'test': df_test
    }

    # Sort each DataFrame by 'exam_dir' and 'filename'
    for key in set_dfs.keys():
        set_dfs[key].sort_values(by=['exam_dir', 'filename'], inplace=True)
        set_dfs[key].reset_index(drop=True, inplace=True)

    # Return the set of DataFrames
    return set_dfs


def main():
    # default yaml file
    DEFAULT_YAML = "configs/efw_tvh_split_v9.yaml"

    # configure the ArgumentParser
    cli_description = 'Split EFW data into train, val, and test sets at given fractions.\n'

    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--yaml', default=DEFAULT_YAML, type=str)

    # extract command line arguments
    args = parser.parse_args()

    # Read feature yaml file
    with open(args.yaml, 'r') as f:
        info = yaml.safe_load(f)

    # look for random seed
    if 'seed' in info:
        np.random.seed(info['seed'])
        random.seed(info['seed'])

    # get source directory root
    root_dir = info['input']['root_dir']

    # get name of metadata folder
    meta_path = os.path.join(root_dir, META_DIR)

    # get name of source data folder
    data_dir = info['input']['data_dir']

    # generate name of distribution folder
    if info['output']['distribution']:
        out_path = os.path.join(info['output']['root_dir'],
                                info['output']['folder'],
                                info['output']['distribution'],
                                info['output']['name'])
    else:
        out_path = os.path.join(info['output']['root_dir'],
                                info['output']['folder'],
                                info['output']['name'])
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    # Get log-normalization columns
    if 'columns' not in info['output']:
        raise ValueError("Log normalization columns not specified.")
    else:
        # The log-normalization columns are specified!
        log_norm_columns = info['output']['columns']
        # Create a dictionary to store mean and std for each column
        mean_std = defaultdict(dict)

    # read the splits dictionary
    splits = info['splits']
    train_fraction = splits['train']['fraction']
    val_fraction = splits['val']['fraction']

    # save console output to file
    sys.stdout = Logger(os.path.join(out_path, f"console.txt"))

    # Read input files
    in_dfs = list()
    for project, filename in info['input']['files'].items():
        this_df = read_spreadsheet_columns(os.path.join(meta_path, filename),
                                           sheet=None,
                                           rows=None,
                                           columns=info['input']['columns'])
        # create partial function with constants set to known values
        outpath_fcn = partial(construct_outpath,
                              root_dir=root_dir,
                              data_dir=data_dir,
                              project=project,
                              ext='.pt')
        # create new column with the full path to the image data
        this_df['outpath'] = this_df.apply(outpath_fcn, axis=1)
        # add the dataframe to the list
        in_dfs.append(this_df)

    # stack all the data frames
    it_df = pd.concat(in_dfs, ignore_index=True)
    # sort by StudyID
    it_df.sort_values(by=['exam_dir', 'filename'], inplace=True)
    # reset the index
    it_df.reset_index(drop=True, inplace=True)

    # Split instance data into train, val, and test sets
    set_dfs = split_by_patient(it_df,
                               train_fraction,
                               val_fraction)

    # Filter DataFrames by tags
    for key, df in set_dfs.items():
        # Get stats prior to filtering
        n_patients = len(df['PID'].unique())
        n_exams = len(df['exam_dir'].unique())
        n_instances = len(df['filename'].unique())
        print(f"Total number of patients: {n_patients},"
              f"                exams: {n_exams},"
              f"                instances: {n_instances}")
        # Get acceptable tags and filter the DataFrame
        print(f"Filtering {key} set by tags")
        valid_tags = get_tag_selection(splits[key]['tags'])
        df = df.loc[np.isin(df['tag'], valid_tags)]
        # Get stats after filtering
        n_patients = len(df['PID'].unique())
        n_exams = len(df['exam_dir'].unique())
        n_instances = len(df['filename'].unique())
        print(f"Final number of patients: {n_patients},"
              f"                exams: {n_exams},"
              f"                instances: {n_instances}")
        #  Apply log normalization
        for col in log_norm_columns:
            # Compute logarithm of column
            fvals = df[col].copy().values.astype(float)
            log_col = np.log(fvals)
            # if training, record the mean and std
            if key == 'train':
                mean = log_col.mean()
                std = log_col.std()
                mean_std[col]['mean'] = mean
                mean_std[col]['std'] = std
            else:
                # if val or test, use mean, std from train
                mean = mean_std[col]['mean']
                std = mean_std[col]['std']
            # Create a new column with the log-normalized values
            df[f"log_{col}"] = (log_col - mean) / std

        # Write the DataFrame to a CSV file
        out_csv = os.path.join(out_path, f"{key}.csv")
        df.to_csv(out_csv, index=False)

    # Print the mean and std dictionary
    print(f"Mean and stdv log values: {mean_std}")

    # close stdout
    sys.stdout.close()


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
