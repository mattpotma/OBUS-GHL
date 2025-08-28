"""
efw_data_selection_v9.py

This script takes an instance table with metadata and filters data according
to two acceptance criteria:
- Biometric and EFW should be consistent across multiple measurements
- (Mean) EFW value should be within a tolerance of Hadlock EFW value
Parameters are specified in a yaml file, as usual.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import yaml
import random
import argparse
import numpy as np
import pandas as pd

from ghlobus.utilities.constants import META_DIR
from ghlobus.utilities.constants import GOOD_VIDEO_MSG
from ghlobus.utilities.constants import EXCLUDE_LABEL
from ghlobus.utilities.constants import UNKNOWN_LABEL
from ghlobus.utilities.data_utils import read_spreadsheet_columns
from ghlobus.utilities.sweep_utils import get_tag_selection
from ghlobus.utilities.biometry_utils import efw_hadlock_4component
from ghlobus.utilities.biometry_utils import ga_hadlock_15


def compute_efw_ga_on_row(row: pd.Series) -> float:
    """
    Compute the Hadlock EFW and GA formlas on a row of the indstance dataframe.
    Args:
        row:       Row of the instance dataframe.

    Returns:
        efwh:     Hadlock EFW value computed on the row.
        gah:      Hadlock GA value computed on the row.
    """
    # Get biometric values from the DataFrame row
    BPD = float(row['BPD'])
    AC = float(row['AC'])
    HC = float(row['HC'])
    FL = float(row['FL'])

    # Compute Hadlock EFW and GA values
    EFW_hadlock = efw_hadlock_4component(BPD, AC, HC, FL)
    GA_hadlock = ga_hadlock_15(BPD, AC, HC, FL)

    return EFW_hadlock, GA_hadlock


def main():
    # Default yaml file
    DEFAULT_YAML = "configs/FAMLI2_efw_data_v9.yaml"

    # Configure the ArgumentParser
    cli_description = "Filter data based on consistent values of "
    cli_description += "biometrics and EFW, as well as EFW that is "
    cli_description += "consistent with Hadlock EFW value."

    # Add arguments to be read from the command line
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--yaml', default=DEFAULT_YAML, type=str)

    # Extract command line arguments
    args = parser.parse_args()

    # read parameter yaml file
    with open(args.yaml, 'r') as f:
        info = yaml.safe_load(f)

    # look for random seed
    if 'seed' in info:
        np.random.seed(info['seed'])
        random.seed(info['seed'])

    # Input and output directory are the same
    in_dir = os.path.join(info['input']['in_dir'], META_DIR)
    in_file = info['it_file']['filename']
    intermediate_file = info['output']['intermediate_file']
    out_file = info['output']['out_file']

    # Check for selection criteria
    if 'selection' not in info:
        raise ValueError("No selection information specified in yaml file.")

    # Check for output filter column
    if 'out_col' not in info['selection']:
        out_col = 'use_for_efw'
    else:
        out_col = info['selection']['out_col']
    # Column that indicates reason for rejection using a code
    # 0 0 0 0 = 0 = valid instance
    # 0 0 0 1 = 1 = instance rejected based on enforced columns
    # 0 0 1 0 = 2 = instance rejected based on tags
    # 0 1 0 0 = 4 = instance rejected based on biometric or EFW consistency
    # 1 0 0 0 = 8 = instance rejected based on Hadlock criteria
    reject_col = 'reject_reason'

    # Column for estimated Hadlock EFW, Hadlock GA
    EFW_hadlock_col = 'EFW_hadlock'
    GA_hadlock_col = 'GA_hadlock'

    # Check for exclusions (TWINS)
    if 'exclude' not in info['selection']:
        exclude_flag = False
        exclude_dir = None
        exclude_file = None
        exclude_col = None
        exclude_values = None
    else:
        # Check for exclusion folder location
        if 'exclude_dir' not in info['selection']['exclude']:
            # No exclusion directory specified
            exclude_flag = False
            exclude_dir = None
            exclude_file = None
            exclude_col = None
            exclude_values = None
            print("Warning: exclusion directory not specified. "
                  + "No exclusions will be applied.")
        else:
            # Exclusion directory is specified
            exclude_dir = info['selection']['exclude']['exclude_dir']
            # Check for exclusion file
            if 'filename' not in info['selection']['exclude']:
                # No exclusion file specified
                exclude_flag = False
                exclude_dir = None
                exclude_file = None
                exclude_col = None
                exclude_values = None
                print("Warning: exclusion filename not specified. "
                      + "No exclusions will be applied.")
            else:
                # Exclusion directory and file are specified
                exclude_flag = True
                exclude_file = info['selection']['exclude']['filename']
                # Check for exclusion column
                if 'column' not in info['selection']['exclude']:
                    # No exclusion column specified, use default
                    exclude_col = 'FINAL_TWIN'
                else:
                    # Exclusion column is specified
                    exclude_col = info['selection']['exclude']['column']
                # Check for exclusion values
                if 'values' not in info['selection']['exclude']:
                    # No exclusion values specified, use default
                    exclude_values = [2, 3, EXCLUDE_LABEL]
                else:
                    # Exclusion values are specified
                    exclude_values = info['selection']['exclude']['values']

    # Check for compulsory columns
    if 'enforce' in info['selection']:
        if info['selection']['enforce']:
            # Check for include column and values
            if 'columns' not in info['selection']['enforce']:
                raise ValueError("No enforce columns specified in yaml file.")
            else:
                enforce_flag = True
                enforce_cols = info['selection']['enforce']['columns']
            # Check for operation on enforce columns
            if 'operation' not in info['selection']['enforce']:
                enforce_operation = 'and'
            else:
                enforce_operation = info['selection']['enforce']['operation']
        else:
            enforce_flag = False
    else:
        enforce_flag = False

    # Check for sweep tag filter
    if 'tags' in info['selection']:
        if info['selection']['tags']:
            tag_flag = True
            if 'column' not in info['selection']['tags']:
                tag_col = 'tag'
            else:
                tag_col = info['selection']['tags']['column']
            # Base case is only blind sweeps
            tags = get_tag_selection(info['selection']['tags'])
        else:
            tag_flag = False
    else:
        tag_flag = False

    # Check for consistency criterion
    if 'consistency' in info['selection']:
        if info['selection']['consistency']:
            # Check for columns to use for consistency
            if 'columns' not in info['selection']['consistency']:
                raise ValueError("No consistency columns specified in yaml file.")
            else:
                consistency_flag = True
                consistency_cols = info['selection']['consistency']['columns']
            if 'threshold' not in info['selection']['consistency']:
                consistency_threshold = 0.1
            else:
                consistency_threshold = info['selection']['consistency']['threshold']
                # Check for operation on enforce columns
                if 'operation' not in info['selection']['consistency']:
                    consistency_operation = 'and'
                else:
                    consistency_operation = info['selection']['consistency']['operation']
        else:
            consistency_flag = False
    else:
        consistency_flag = False

    # Check for Hadlock criteria
    if 'hadlock_threshold' in info['selection']:
        hadlock_flag = True
        hadlock_threshold = info['selection']['hadlock_threshold']
    else:
        hadlock_flag = False

    # Read the instance table
    it_file_path = os.path.join(
        in_dir,
        in_file)
    rows = None
    columns = info['it_file']['columns']
    if 'rows' in info['it_file']:
        rows = info['it_file']['rows']
    it_df = read_spreadsheet_columns(it_file_path,
                                     rows=rows,
                                     columns=columns)
    print(f"Read {it_df.shape[0]} instances from {in_file}.")

    # Only use "good_videos" for processing
    it_df = it_df[it_df['fail_reason'] == GOOD_VIDEO_MSG]
    it_df.reset_index(drop=True, inplace=True)
    print(f"Of these, {it_df.shape[0]} instances are {GOOD_VIDEO_MSG}.")

    # Check patients against exclude list
    if exclude_flag:
        # Read the exclusion file and force set these values
        exclude_file_path = os.path.join(
            exclude_dir,
            exclude_file)
        exclude_df = read_spreadsheet_columns(exclude_file_path, columns=['PID', exclude_col])
        # Filter on exclude_col values;
        # If exclude_col is one of these, exclude these patients
        exclusions = exclude_df[exclude_col].isin(exclude_values)
        exclude_df = exclude_df[~exclusions]
        exclude_df.reset_index(drop=True, inplace=True)
        exclude_pids = exclude_df['PID'].values

        message = f"Read {exclude_df.shape[0]} PIDs from exclusion file "
        message += f"{exclude_file}."
        print(message)

        # exclude videos with PIDs in the exclusion list
        PIDs = it_df['PID'].values
        exclude = np.isin(PIDs, exclude_pids)
        # count PIDs, instances that were rejected based on exclued PID list
        rejected_pids = PIDs[exclude]
        n_rejected_pids = len(np.unique(rejected_pids))
        n_rejected_instances = exclude.sum()
        if any(exclude):
            it_df = it_df[~exclude]
            it_df.reset_index(drop=True, inplace=True)
        print(f"Excluded {n_rejected_pids} patients with excluded PIDs.")
        print(f"Excluded {n_rejected_instances} instances with excluded PIDs.")
        print(f"There are {it_df.shape[0]} remaining eligible instances.")

    # New columns for the output instance file
    it_df[out_col] = 1
    it_df[reject_col] = 0
    it_df['EFW_hadlock'] = 0
    it_df['GA_hadlock'] = 0
    it_df = it_df.astype({
        out_col: 'int64',
        reject_col: 'int64',
        EFW_hadlock_col: 'float64',
        GA_hadlock_col: 'float64',
    })

    # Are any columns enforced to have values?
    if enforce_flag:
        # Compute enforce mask from selection criteria
        valid_enforce_masks = list()
        # noinspection PyUnboundLocalVariable
        for enforce_col in enforce_cols:
            # check if we have an enforce entry where
            # at least one of the columns must be non-empty
            empty = it_df[enforce_col] == ''
            na = pd.isna(it_df[enforce_col])
            null = pd.isnull(it_df[enforce_col])
            valid_enforce_masks.append(~empty & ~na & ~null)
        # noinspection PyUnboundLocalVariable
        if enforce_operation == 'or':
            valid_enforce_mask = np.logical_or.reduce(valid_enforce_masks)
        else:
            valid_enforce_mask = np.logical_and.reduce(valid_enforce_masks)

        # Set the enforced instances to exclude label (0)
        invalid_enforce_mask = ~valid_enforce_mask
        print(f"Removing {invalid_enforce_mask.sum()} instances.")
        print("--based on missing values in enforced columns.")
        # Filter the data based on the enforce mask
        it_df.loc[invalid_enforce_mask, out_col] = 0
        it_df.loc[invalid_enforce_mask, reject_col] += 1
    else:
        # No enforced columns, so all instances are valid
        valid_enforce_mask = np.ones(it_df.shape[0], dtype=bool)
    # Remove instances without enforced columns
    it_df = it_df.loc[valid_enforce_mask]

    # Are tag values being filtered?
    if tag_flag:
        # Compute tag mask from selection criteria
        # noinspection PyUnboundLocalVariable
        valid_tag_mask = np.isin(it_df[tag_col], tags)
        # Set the enforced instances to exclude label
        invalid_tag_mask = ~valid_tag_mask
        print(f"Removing {invalid_tag_mask.sum()} instances.")
        print("--based on disallowed sweep tags.")
        # Filter the data based on tag selection
        it_df.loc[invalid_tag_mask, out_col] = 0
        it_df.loc[invalid_tag_mask, reject_col] += 2
    else:
        # No tag filtering, so all instances are valid
        valid_tag_mask = np.ones(it_df.shape[0], dtype=bool)

    # Check for rejects based on biometric consistency
    if consistency_flag:
        # Compute consistency mask from selection criteria
        # noinspection PyUnboundLocalVariable
        valid_consistency_masks = list()
        for consistency_col in consistency_cols:
            mask = it_df[consistency_col].astype(float) <= consistency_threshold
            valid_consistency_masks.append(mask)
        # noinspection PyUnboundLocalVariable
        if consistency_operation == 'or':
            valid_consistency_mask = np.logical_or.reduce(valid_consistency_masks)
        else:
            valid_consistency_mask = np.logical_and.reduce(valid_consistency_masks)

        # Set the enforced instances to exclude label (0)
        invalid_consistency_mask = ~valid_consistency_mask
        print(f"Removing {invalid_consistency_mask.sum()} instances.")
        print("--based on inconsistent biometrics or EFW.")
        # Filter the data based on the consistency criteria
        it_df.loc[invalid_consistency_mask, out_col] = 0
        it_df.loc[invalid_consistency_mask, reject_col] += 4
    else:
        # No consistency filtering, so all instances are valid
        valid_consistency_mask = np.ones(it_df.shape[0], dtype=bool)

    # Check for Hadlock criteria
    if hadlock_flag:
        # Compute the Hadlock relative error
        EFW = it_df['EFW'].astype(float).values
        efw_ga_values =it_df.apply(compute_efw_ga_on_row, axis=1)
        EFW_hadlock = np.array([s[0] for s in efw_ga_values])
        GA_hadlock = np.array([s[1] for s in efw_ga_values])
        rel_err = np.abs(EFW - EFW_hadlock) / EFW
        # Compute Hadlock mask from selection criteria
        valid_hadlock_mask = rel_err <= hadlock_threshold
        # Set the hadlock filtered instances to exclude label (0)
        invalid_hadlock_mask = ~valid_hadlock_mask
        print(f"Removing {invalid_hadlock_mask.sum()} instances.")
        print("--based on Hadlock EFW consistency criteria.")
        # Filter the data based on the Hadlock criteria
        it_df.loc[invalid_hadlock_mask, out_col] = 0
        it_df.loc[invalid_hadlock_mask, reject_col] += 8
        # Populate Hadlock EFW and GA columns
        it_df[EFW_hadlock_col] = EFW_hadlock
        it_df[GA_hadlock_col] = GA_hadlock
    else:
        # No Hadlock filtering, so all instances are valid
        valid_hadlock_mask = np.ones(it_df.shape[0], dtype=bool)

    # Save intermediate result
    inter_file_path = os.path.join(
        in_dir,
        intermediate_file)
    it_df.to_csv(inter_file_path, index=False)
    message = f"Saved intermediate file with {it_df.shape[0]} "
    message += f"instances to {intermediate_file}."
    print(message)

    # Remove the invalid instances
    valid_instance_mask = np.logical_and(
        valid_tag_mask,
        valid_consistency_mask,
        valid_hadlock_mask,
    )
    it_df = it_df.loc[valid_instance_mask]

    # Determine the number of unique patients
    n_patients = it_df['PID'].nunique()
    print(f"Selected {n_patients} unique patients.")
    n_exams = it_df['exam_dir'].nunique()
    print(f"Selected {n_exams} unique exams.")
    # Number of instances
    n_videos = valid_instance_mask.sum()
    print(f"Selected {n_videos} valid videos.")

    # Save final result
    out_file_path = os.path.join(
        in_dir,
        out_file)
    it_df.to_csv(out_file_path, index=False)
    print(f"Saved selected data to {out_file}.")


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
