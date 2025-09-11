"""
merge_sr_crf_v9.py

This script combines the structured_reports that were previously converted to
tabular form and the case report forms to create a single data table containing
all the relevant information.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd

from tablify_sr_v9 import get_tabular_dir
from ghlobus.utilities.data_utils import read_spreadsheet
from ghlobus.utilities.data_utils import read_spreadsheet_columns
from ghlobus.utilities.constants import META_DIR


def us_lie_integer(us_lie: str) -> int:
    """
    Convert a string representing a fetal presentation to an integer.
    Use the following mapping:
    N/A: -88
    Cephalic:
    Breech: 2
    Transverse: 3
    Variable / NA: 4
    Oblique: 5

    Args:
        us_lie (str): A string representing the us_lie .

    Returns:
        int: The integer value of the string.
    """
    if us_lie == '':
        return -88
    elif us_lie == 'Cephalic':
        return 1
    elif us_lie == 'Breech':
        return 2
    elif us_lie == 'Transverse':
        return 3
    elif us_lie == 'Variable / NA':
        return 4
    elif us_lie == 'Oblique':
        return 5
    else:
        raise ValueError(f"Invalid us_lie value: {us_lie}")


def remove_visit(studyid: str) -> str:
    """
    When PID has no value, get from StudyID

    Args:
        studyid:  The StudyID

    Returns:
        str:      The extracted PID

    """
    # split by dash
    splits = studyid.split('-')
    # remove the entity (integer) after the last dash
    pid = "-".join(splits[:-1])
    return pid


def main():
    # Default yaml file
    DEFAULT_YAML = "configs/FAMLI2_sr_crf.yaml"

    # Configure the ArgumentParser
    cli_description = "Combine case report forms and structured reports metadata."

    # Add arguments to be read from the command line
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--yaml', default=DEFAULT_YAML, type=str)

    # Extract command line arguments
    args = parser.parse_args()

    # read parameter yaml file
    with open(args.yaml, 'r') as f:
        info = yaml.safe_load(f)

    # Create output directory if not there
    out_dir = os.path.join(info['output']['out_dir'], META_DIR)
    if not os.path.exists(out_dir):
        raise ValueError(f"Output directory {out_dir} does not exist.")

    # Frequently used directories, params
    raw_dir = info['input']['raw_dir']
    project = info['input']['project']

    # get name of Tabular_data directory
    tabular_dir = get_tabular_dir(raw_dir, project)

    # get name of CRF directory
    crf_dir = os.path.join(raw_dir,  project, tabular_dir)

    # the SR files are output from tablify_sr_v9.py step
    sr_files = info['input']['sr_files']

    # check if required entries are in the yaml file
    if 'crf_sheet' not in info['input']:
        crf_sheet = None
    else:
        crf_sheet = info['input']['crf_sheet']

    # see if there are structured report files
    num_sr_files = 0
    if 'sr_files' in info['input']:
        num_sr_files = len(sr_files)
    if num_sr_files == 0:
        raise ValueError("No structured report files specified in yaml file.")

    # cycle through the tabular structured report files
    sr_dfs = list()
    for sr_file in sr_files:
        sr_path = os.path.join(
            out_dir,
            sr_file)
        sr_df = read_spreadsheet(sr_path)
        print(f"Read {sr_df.shape[0]} rows from {sr_file}.")
        sr_dfs.append(sr_df)

    # merge the structured report dataframes (if more than one)
    sr_df = sr_dfs[0]
    print(f"Started with first structured report file {sr_files[0]}.")
    for i in range(1, num_sr_files):
        sr_df = sr_df.merge(sr_dfs[i], how='outer').reset_index(drop=True).fillna('')
        print(f"Merged structured report file {sr_files[i]}.")
    sr_df = sr_df.fillna('')

    # read the case report form data
    crf_path = os.path.join(
        crf_dir,
        info['input']['crf_file'])
    crf_df = read_spreadsheet_columns(crf_path,
                                      sheet=crf_sheet,
                                      rows=None,
                                      columns=info['crf_dict']['columns'])
    crf_df = crf_df.fillna('')
    print(f"Read {crf_df.shape[0]} rows from {crf_path}.")

    # finds exams that are ineligible
    if 'eligible' in crf_df.columns:
        eligible = np.logical_or(crf_df['eligible'] == 'Eligible', crf_df['eligible'] == '1')
        # crf_df = crf_df[eligible]
        print(f"Exams {(~eligible).sum()} that are ineligible.")

    # rename (some) columns
    if 'rename_columns' in info['crf_dict']:
        print(f"Renaming columns in CRF table")
        crf_df.rename(columns=info['crf_dict']['rename_columns'], inplace=True)

    # transform (some) columns
    if 'transform_columns' in info['crf_dict']:
        for col, func in info['crf_dict']['transform_columns'].items():
            if func == 'us_lie_integer':
                # for FAMLI2 dataset, us_lie must be converted from text to numeric
                # e.g. 'Cephalic' to 1
                crf_df[col] = crf_df[col].apply(us_lie_integer)
            elif func == 'StudyID':
                # DXA has only one exam per patient and the CRF has no StudyID column
                # The PID column is copied as StudyID and "-1" is appended for StudyID
                crf_df[func] = crf_df[col] + '-1'
            else:
                raise ValueError(f"Invalid transform function: {func}")

    # merge the structured report and case report form dataframes
    out_df = sr_df.merge(
        crf_df,
        on='StudyID',
        how='outer',
        indicator=True).reset_index(drop=True)
    print(f"Merged structured report and case report form dataframes.")

    # rename '_merge' column
    out_df.rename(columns={'_merge': 'found_in'}, inplace=True)

    # rename "left_only" to "sr_only"
    out_df['found_in'] = out_df['found_in'].astype(str)
    left_only = np.array(out_df['found_in'] == 'left_only')
    out_df.loc[left_only, 'found_in'] = 'sr_only'
    # rename "right_only" to "crf_only"
    right_only = np.array(out_df['found_in'] == 'right_only')
    out_df.loc[right_only, 'found_in'] = 'crf_only'
    # crf_only exams are missing PID (PID_y)
    out_df.loc[right_only, 'PID_y'] = out_df.loc[right_only, 'StudyID'].apply(remove_visit)

    # rename PID_x to PID
    out_df.rename(columns={'PID_x': 'PID'}, inplace=True)
    # if PID_x is missing, get value from PID_y and remove column

    empty = out_df['PID'] == ''
    na = pd.isna(out_df['PID'])
    null = pd.isnull(out_df['PID'] == '')
    missing_PID_x = np.logical_or.reduce((empty, na, null))
    out_df.loc[missing_PID_x, 'PID'] = out_df.loc[missing_PID_x, 'PID_y']
    out_df.drop(columns=['PID_y'], inplace=True)
    # move PID column to beginning
    cols = out_df.columns.tolist()
    pid_idx = cols.index('PID')
    cols = ['PID'] + cols[:pid_idx] + cols[pid_idx + 1:]
    out_df = out_df[cols]

    # replace empty Manufacturer string with "Unknown"
    out_df['Manufacturer'] = out_df['Manufacturer'].replace(np.nan, -99)
    out_df.loc[out_df['Manufacturer'] == -99, 'Manufacturer'] = 'Unknown'

    # replace empty Model string with "Unnamed"
    out_df['ManufacturerModelName'] = out_df['ManufacturerModelName'].replace(np.nan, -99)
    out_df.loc[out_df['ManufacturerModelName'] == -99, 'ManufacturerModelName'] = 'Unnamed'

    # save the final exam metadata file
    out_path = os.path.join(
        out_dir,
        info['output']['out_file'])
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_df.shape[0]} exams data to {out_path}.")


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
