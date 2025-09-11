"""
twin_data_selection_v9.py

This script takes an instance table with metadata and creates a subset containing
all the twin/triplet data as well as a subset containing only the single data.
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


def main():
    # Default yaml file
    DEFAULT_YAML = "configs/FAMLI2_twin_data_v9.yaml"

    # Configure the ArgumentParser
    cli_description = "Select subset of data with all twins, some singles."

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

    # Check for selection criteria
    if 'selection' not in info:
        raise ValueError("No selection information specified in yaml file.")
    # Check for output column
    if 'out_col' not in info['selection']:
        out_col = 'TWIN'
    else:
        out_col = info['selection']['out_col']

    # Check for twins criteria
    if 'twins' not in info['selection']:
        raise ValueError("No twins criteria specified in yaml file.")
    if 'patients' not in info['selection']['twins']:
        target_twin_patients = None
    else:
        target_twin_patients = info['selection']['twins']['patients']
    # Check for include column and values
    if 'columns' not in info['selection']['twins']['include']:
        raise ValueError("No twins include columns specified in yaml file.")
    else:
        twin_cols = info['selection']['twins']['include']['columns']
    if 'values' not in info['selection']['twins']['include']:
        raise ValueError("No twins include values specified in yaml file.")
    else:
        twin_include_vals = info['selection']['twins']['include']['values']
    # Check for operation on twins columns
    if 'operation' not in info['selection']['twins']['include']:
        twin_operation = 'and'
    else:
        twin_operation = info['selection']['twins']['include']['operation']
    # Check for twins label
    if 'label' not in info['selection']['twins']:
        twin_label = 1
    else:
        twin_label = info['selection']['twins']['label']

    # Check for singles criteria
    if 'singles' not in info['selection']:
        raise ValueError("No singles criteria specified in yaml file.")
    if 'patients' not in info['selection']['singles']:
        target_single_patients = None
    else:
        target_single_patients = info['selection']['singles']['patients']
    # Check for singles label
    if 'label' not in info['selection']['singles']:
        single_label = 0
    else:
        single_label = info['selection']['singles']['label']
    # Check for compulsory columns
    if 'enforce' in info['selection']['singles']:
        if info['selection']['singles']['enforce']:
            # Check for include column and values
            if 'columns' not in info['selection']['singles']['enforce']:
                raise ValueError("No enforce columns specified in yaml file.")
            else:
                enforce_flag = True
                enforce_cols = info['selection']['singles']['enforce']['columns']
            # check if multiple columns are allowed for each enforced columns
            for ind, ec in enumerate(enforce_cols):
                if ',' in ec:
                    enforce_cols[ind] = ec.replace(' ', '').split(',')
            # Check for operation on enforce columns
            if 'operation' not in info['selection']['singles']['enforce']:
                enforce_operation = 'and'
            else:
                enforce_operation = info['selection']['singles']['enforce']['operation']
        else:
            enforce_flag = False
    else:
        enforce_flag = False
    # Check for sweep tag filter
    if 'tags' in info['selection']['singles']:
        if info['selection']['singles']['tags']:
            tag_flag = True
            if 'column' not in info['selection']['singles']['tags']:
                tag_col = 'tag'
            else:
                tag_col = info['selection']['singles']['tags']['column']
            # Base case is only blind sweeps
            tags = get_tag_selection(info['selection']['singles']['tags'])
        else:
            tag_flag = False
    else:
        tag_flag = False

    # Check for exceptions
    if 'exception' not in info['selection']:
        exception_flag = False
        except_dir = None
        except_file = None
        exclude_label = EXCLUDE_LABEL
    else:
        # Check for exception folder location
        if 'except_dir' not in info['selection']['exception']:
            # No exception directory specified
            exception_flag = False
            except_dir = None
            except_file = None
            exclude_label = EXCLUDE_LABEL
            print("Warning: exception directory not specified. "
                  + "No exceptions will be applied.")
        else:
            # Exception directory is specified
            except_dir = info['selection']['exception']['except_dir']
            # Check for exception file
            if 'filename' not in info['selection']['exception']:
                # No exception file specified
                exception_flag = False
                except_dir = None
                except_file = None
                exclude_label = EXCLUDE_LABEL
                print("Warning: exception filename not specified. "
                      + "No exceptions will be applied.")
            else:
                # Exception directory and file are specified
                exception_flag = True
                except_file = info['selection']['exception']['filename']
                # Check for exception label
                if 'exclude_label' not in info['selection']['exception']:
                    exclude_label = EXCLUDE_LABEL
                else:
                    exclude_label = info['selection']['exception']['exclude_label']

    # Read the instance table
    it_file_path = os.path.join(
        in_dir,
        info['it_file']['filename'])
    rows = None
    if 'rows' in info['it_file']:
        rows = info['it_file']['rows']
    it_df = read_spreadsheet_columns(
        it_file_path,
        rows=rows,
        columns=info['it_file']['columns'])
    message = f"Read {it_df.shape[0]} instances from "
    message += f"{info['it_file']['filename']}."
    print(message)

    # Only use "good_videos" for processing
    it_df = it_df[it_df['fail_reason'] == GOOD_VIDEO_MSG]
    it_df.reset_index(drop=True, inplace=True)
    print(f"Of these, {it_df.shape[0]} instances are {GOOD_VIDEO_MSG}.")

    # convert twins columns to integer
    for twin_col in twin_cols:
        # replace empty strings with unknown label
        it_df[twin_col] = it_df[twin_col].replace('', str(UNKNOWN_LABEL))
        it_df[twin_col] = it_df[twin_col].astype(float).astype(int)

    # Check patients against exception list
    if exception_flag:
        # Read the exception file and force set these values
        except_file_path = os.path.join(
            except_dir,
            except_file)
        except_df = read_spreadsheet_columns(except_file_path, columns=['PID']+twin_cols)
        message = f"Read {except_df.shape[0]} PIDs from exception file "
        message += f"{except_file}."
        print(message)

        # convert twin columns to integer
        for twin_col in twin_cols:
            # replace empty strings with exclude label
            except_df[twin_col] = except_df[twin_col].replace('', str(exclude_label))
            except_df[twin_col] = except_df[twin_col].astype(float).astype(int)

        # noinspection PyUnboundLocalVariable
        for pid, val0, val1 in except_df[['PID'] + twin_cols].values:
            this_pt = it_df['PID'].values == pid
            if any(this_pt):
                it_df.loc[this_pt, twin_cols[0]] = val0
                it_df.loc[this_pt, twin_cols[1]] = val1

        # exclude videos with exclude label
        # noinspection PyUnboundLocalVariable
        exclude = np.array(it_df[twin_cols[0]].values == exclude_label)
        n_excluded = exclude.sum()
        if any(exclude):
            it_df = it_df[~exclude]
            it_df.reset_index(drop=True, inplace=True)
        print(f"Excluded {n_excluded} instances with exclude label.")
        print(f"There are {it_df.shape[0]} remaining eligible instances.")

    # Compute twin mask from selection criteria
    twin_masks = list()
    for twin_col in twin_cols:
        twin_masks.append(it_df[twin_col].isin(twin_include_vals))
    if twin_operation == 'or':
        twin_mask = np.logical_or.reduce(twin_masks)
    else:
        twin_mask = np.logical_and.reduce(twin_masks)

    # Everything that's not a twin is assumed to be single
    # unless it's specifically excluded as above
    single_mask = ~twin_mask
    print(f"There are {single_mask.sum()} singlet instances.")
    print(f"There are {twin_mask.sum()} twin instances.")

    # Set the twin and single labels
    it_df[out_col] = 0
    it_df = it_df.astype({out_col: 'int64'})
    it_df.loc[twin_mask, out_col] = twin_label
    it_df.loc[single_mask, out_col] = single_label

    # Are any columns enforced to have values?
    if enforce_flag:
        # Compute enforce mask from selection criteria
        valid_enforce_masks = list()
        # noinspection PyUnboundLocalVariable
        for enforce_col in enforce_cols:
            # check if we have an enforce entry where
            # at least one of the columns must be non-empty
            if isinstance(enforce_col, list):
                this_ec_masks = list()
                for ec in enforce_col:
                    empty = it_df[ec] == ''
                    na = pd.isna(it_df[ec])
                    null = pd.isnull(it_df[ec])
                    this_ec_masks.append(~empty & ~na & ~null)
                valid_enforce_masks.append(np.logical_or.reduce(this_ec_masks))
            else:
                empty = it_df[enforce_col] == ''
                na = pd.isna(it_df[enforce_col])
                null = pd.isnull(it_df[enforce_col])
                valid_enforce_masks.append(~empty & ~na & ~null)
        # noinspection PyUnboundLocalVariable
        if enforce_operation == 'or':
            valid_enforce_mask = np.logical_or.reduce(valid_enforce_masks)
        else:
            valid_enforce_mask = np.logical_and.reduce(valid_enforce_masks)

        # Set the enforced singlet instances to exclude label
        # enforced columns are not applied to twins
        invalid_mask = np.logical_and(single_mask, ~valid_enforce_mask)
        valid_enforce_single_mask = np.logical_and(single_mask, valid_enforce_mask)
        print(f"Removing {invalid_mask.sum()} singlet instances.")
        print("--based on missing values in enforced columns.")
        # Filter the data based on the twin and single masks
        it_df.loc[invalid_mask, out_col] = exclude_label
    else:
        valid_enforce_single_mask = single_mask

    # Are tag values being filtered?
    if tag_flag:
        # Compute tag mask from selection criteria
        # noinspection PyUnboundLocalVariable
        valid_tag_mask = np.isin(it_df[tag_col], tags)
        # Set the enforced single instances to exclude label
        invalid_mask = np.logical_and(single_mask, ~valid_tag_mask)
        valid_tag_single_mask = np.logical_and(single_mask, valid_tag_mask)
        print(f"Removing {invalid_mask.sum()} singlet instances.")
        print("--based on disallowed sweep tags.")
        # Filter the data based on the twin and single masks
        it_df.loc[invalid_mask, out_col] = exclude_label
    else:
        valid_tag_single_mask = single_mask

    # Save intermediate result
    inter_file_path = os.path.join(
        in_dir,
        info['output']['intermediate_file'])
    it_df.to_csv(inter_file_path, index=False)
    message = f"Saved intermediate file with {it_df.shape[0]} "
    message += f"instances to {info['output']['intermediate_file']}."
    print(message)

    # Remove the invalid singleton instances
    valid_instance = np.logical_or(
        twin_mask,
        np.logical_and(valid_enforce_single_mask,
                       valid_tag_single_mask))
    it_df = it_df[valid_instance]
    twin_mask = twin_mask[valid_instance]
    single_mask = single_mask[valid_instance]

    # Determine the number of unique twin and single patients
    n_twin_patients = it_df['PID'][twin_mask].nunique()
    n_single_patients = it_df['PID'][single_mask].nunique()
    print(f"Found {n_twin_patients} unique twin patients.")
    print(f"Found {n_single_patients} unique singleton patients.")

    # Sample the twin data?
    if target_twin_patients is not None:
        # Sample the twin data
        twin_patients = it_df['PID'][twin_mask].unique()
        twin_patients = np.random.choice(twin_patients, target_twin_patients, replace=False)
        twin_mask = it_df['PID'].isin(twin_patients)
        print(f"Sampled {target_twin_patients} unique twin patients.")

    # Sample the single data?
    if target_single_patients is not None:
        # Sample the single data
        single_patients = it_df['PID'][single_mask].unique()
        single_patients = np.random.choice(
            single_patients,
            min(n_single_patients, target_single_patients),
            replace=False)
        single_mask = it_df['PID'].isin(single_patients)
        print(f"Sampled {target_single_patients} unique singleton patients.")

    # Apply the subsample
    sampled_instances = np.logical_or(twin_mask, single_mask)
    it_df = it_df[sampled_instances]

    # Number of selected unique twin and single video
    n_twin_videos = twin_mask.sum()
    n_single_videos = single_mask.sum()
    print(f"Selected {n_twin_videos} unique twin videos.")
    print(f"Selected {n_single_videos} unique singleton videos.")

    # Save final result
    out_file_path = os.path.join(
        in_dir,
        info['output']['out_file'])
    it_df.to_csv(out_file_path, index=False)
    print(f"Saved selected data to {info['output']['out_file']}.")


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
