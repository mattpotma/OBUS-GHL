"""
merge_tag_v9.py

This script merges corrected sweep tag information into the prototype instance table.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd

from ghlobus.utilities.data_utils import read_spreadsheet
from ghlobus.utilities.data_utils import read_spreadsheet_columns
from ghlobus.utilities.constants import META_DIR


def remove_empty_tags(tag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the corrected tag data such that empty tags are removed.

    Parameters:
        tag_df (pd.DataFrame):  The LLM tag data frame
    Returns:
         pd.DataFrame:          The LLM tag table correspond to
                                empty tags in the instance table
    """
    # remove empty tags from tag_df
    tag_df = tag_df[tag_df['tag'].notnull()]
    # remove Unknown tags from tag_df
    tag_df = tag_df[tag_df['tag'] != "Unknown"]

    # return the tag table with empty and unknown tags removed
    return tag_df


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    # Default yaml file
    DEFAULT_YAML = "configs/FAMLI2_tag_v9.yaml"

    # Configure the ArgumentParser
    cli_description = "Merge tag sweep information into prototype file."

    # Add arguments to be read from the command line
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--yaml', default=DEFAULT_YAML, type=str)

    # Extract command line arguments
    args = parser.parse_args()

    # read parameter yaml file
    with open(args.yaml, 'r') as f:
        info = yaml.safe_load(f)

    # get name of metadata directory
    meta_path = os.path.join(info['input']['root_dir'], META_DIR)

    # get name of corrections directory
    corr_path = os.path.join(info['correction']['except_dir'])

    # read the prototype file
    proto_file_path = os.path.join(meta_path, info['input']['filename'])
    it_df = read_spreadsheet(proto_file_path)
    print(f"Read {it_df.shape[0]} rows from {info['input']['filename']}.")

    # read the corrected tag file
    tag_file_path = os.path.join(corr_path, info['correction']['filename'])
    tag_df = read_spreadsheet_columns(
        tag_file_path,
        columns=info['correction']['columns'])
    print(f"Read {tag_df.shape[0]} rows from {info['correction']['filename']}.")

    # see if any columns need to be renamed
    if 'rename_columns' in info['correction']:
        print(f"Renaming columns in tag table")
        tag_df.rename(columns=info['correction']['rename_columns'], inplace=True)

    # set the indices for update
    tag_df.set_index('filename', inplace=True)
    it_df.set_index('filename', inplace=True)
    # update the tag data from the corrected LLM tag table
    it_df.update(tag_df)
    # reset the index (DO NOT drop, it is filename!)
    it_df.reset_index(drop=False, inplace=True)

    # replace empty tags with unknown
    empty = np.array(it_df['tag'] == "")
    it_df.loc[empty, 'tag'] = "Unknown"

    # move columns into the right order
    cols = it_df.columns.tolist()
    desired_initial_cols = ['PID', 'StudyID', 'relpath', 'StudyDate', 'filename']
    for col in reversed(desired_initial_cols):
        cols.remove(col)
        cols.insert(0, col)
        it_df = it_df[cols]

    # save the final exam metadata file
    out_file_path = os.path.join(
        meta_path,
        info['output']['filename'])
    it_df.to_csv(out_file_path, index=False)
    print(f"Saved prototype_tag data to {info['output']['filename']}.")
