"""
merge_instance_exam_v9.py

Combine instance and exm data into an instance table with selected metadata.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from functools import partial
from typing import List

from tablify_sr_v9 import get_tabular_dir
from ghlobus.utilities.data_utils import empty_cells
from ghlobus.utilities.data_utils import read_spreadsheet
from ghlobus.utilities.data_utils import read_spreadsheet_columns
from ghlobus.utilities.constants import MP4_FILE_TYPE
from ghlobus.utilities.constants import META_DIR


def make_rel_path(pid_date_time: str, project: str) -> str:
    """
    Construct a relative path from the project and pid_date_time string.

    Args:
        project (str): The root directory (name) for the project.
        pid_date_time (str): Date time string that is name of data directory.

    Returns:
        str: The relative path constructed from project and pid_date_time.
    """
    # extract year and month from pid_date_time
    parts = pid_date_time.split('_')
    study_id, year_month_day, time = parts
    year = year_month_day[:4]
    month = year_month_day[4:6]

    # form the relative path to data directory
    rel_path = os.path.join(project, 'Ultrasound', f"{year}-{month}", pid_date_time)

    return rel_path


def create_join_col(df: pd.DataFrame,
                    join_columns: list,
                    sep: str = '_',
                    ) -> pd.Series:
    """
    Create a join column from a list of columns in the dataframe.

    Parameters
    ----------
        df: pd.DataFrame     pandas data frame from which to construct join column
        join_columns: list   column names to join
        sep: str             separator to use between column values

    Returns
    -------
        pd.Series:           series of join column values
    """
    join_col_val = df[join_columns[0]].astype(str)
    for col in join_columns[1:]:
        join_col_val += sep + df[col].astype(str)

    return join_col_val


def drop_duplicates_on_join_column(df: pd.DataFrame,
                                   join_columns: list,
                                   sep: str = '_',
                                   ) -> pd.DataFrame:
    """
    Drop duplicate rows based on a list of columns in the dataframe.

    Parameters
    ----------
        df: pd.DataFrame     pandas data frame from which to construct join column
        join_columns: list   column names to join
        sep: str             separator to use between column values

    Returns
    -------
        df:                  Dataframe of join column values
    """
    # create the join column and add it to the dataframe
    df['join_col'] = create_join_col(df, join_columns, sep)
    # drop duplicates based on the join column
    df.drop_duplicates(subset='join_col', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # now drop the join column
    df.drop(columns=['join_col'], inplace=True)

    return df


def priority_sort_indices(matches: np.ndarray,
                          indices: np.ndarray,
                          priority: List[str],
                          ) -> np.ndarray:
    """
    Sorts indices along with associated strings, according to a priority list of strings.
    Args:
        matches:    list of current matched strings
        indices:    indices associated with strings
        priority:   priority order of strings

    Returns:
        np.ndarray: indices sorted by priority
    """
    # priority of strings in numeric form
    priority_numeric = np.arange(len(priority), dtype=int)
    # numeric priority assigned matches
    matches_numeric = np.array([priority_numeric[priority.index(x)] for x in matches])
    # find the sort order of the matches
    order = np.argsort(matches_numeric)
    # sort the indices by priority
    sorted_indices = indices[order]
    return sorted_indices


def main():
    # Default yaml file
    DEFAULT_YAML = "configs/FAMLI2_IT_exam_v9.yaml"

    # priority for source of exam metadata for study matches
    match_priority = ['both', 'sr_only', 'crf_only']

    # Configure the ArgumentParser
    cli_description = "Combine instance and exam data to a "
    cli_description += "metadata-enhanced instance table."

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
        os.makedirs(out_dir, exist_ok=True)

    # Frequently used values
    raw_dir = info['input']['raw_dir']
    project = info['input']['project']

    # get name of Tabular_data directory
    tabular_dir = get_tabular_dir(raw_dir, project)

    # get name of instance directory
    inst_dir = os.path.join(raw_dir, project, tabular_dir)

    # read main instance table
    it_path = os.path.join(
        inst_dir,
        info['main_IT_file']['filename'])
    rows = None
    if 'rows' in info['main_IT_file']:
        rows = info['main_IT_file']['rows']
    it_df = read_spreadsheet_columns(it_path,
                                     sheet=None,
                                     rows=rows,
                                     columns=info['main_IT_file']['columns'])
    print(f"Read {it_df.shape[0]} rows from {it_path}.")

    # rename (some) columns
    if 'rename_columns' in info['main_IT_file']:
        print(f"Renaming columns in main instance table")
        it_df.rename(columns=info['main_IT_file']['rename_columns'], inplace=True)

    # drop rows with empty filename
    n_orig = it_df.shape[0]
    it_df = it_df[it_df['filename'] != ''].reset_index(drop=True)
    n_final = it_df.shape[0]
    print(f"Dropped {n_orig - n_final} rows with missing filenames.")

    # drop rows with duplicate filenames
    if 'duplicate_columns' in info['main_IT_file']:
        n_orig = it_df.shape[0]
        it_df = drop_duplicates_on_join_column(it_df, info['main_IT_file']['duplicate_columns'])
        n_final = it_df.shape[0]
        print(f"Dropped {n_orig - n_final} rows with duplicate filenames.")

    # Manufacturer renames
    if 'manufacturer_map' in info:
        print(f"Renaming Manufacturer names")
        for key, text in info['manufacturer_map'].items():
            it_df['Manufacturer'][it_df['Manufacturer'].str.contains(text)] = key

    # ManufacturerModelName renames
    if 'manufacturer_model_name_map' in info:
        print(f"Renaming Manufacturer Model names")
        for key, text in info['manufacturer_model_name_map'].items():
            it_df['ManufacturerModelName'][it_df['ManufacturerModelName'] == text] = key

    # replace Butterfly with empty Model string with "Unnamed"
    it_df['ManufacturerModelName'][
        np.logical_and(
            it_df['Manufacturer'] == 'Butterfly',
            it_df['ManufacturerModelName'] == '')] \
        = 'Unnamed'

    # read Butterfly instance table
    mp4_it_path = os.path.join(
        inst_dir,
        info['mp4_IT_file']['filename'])
    rows = None
    if 'rows' in info['mp4_IT_file']:
        rows = info['mp4_IT_file']['rows']
    mp4_it_df = read_spreadsheet_columns(mp4_it_path,
                                         sheet=None,
                                         rows=rows,
                                         columns=info['mp4_IT_file']['columns'])
    print(f"Read {mp4_it_df.shape[0]} rows from {mp4_it_path}.")

    # rename (some) columns
    if 'rename_columns' in info['mp4_IT_file']:
        print(f"Renaming columns in mp4 instance table")
        mp4_it_df.rename(columns=info['mp4_IT_file']['rename_columns'], inplace=True)

    # drop rows with duplicate filenames
    if 'duplicate_columns' in info['mp4_IT_file']:
        n_orig = mp4_it_df.shape[0]
        mp4_it_df = drop_duplicates_on_join_column(mp4_it_df, info['mp4_IT_file']['duplicate_columns'])
        n_final = mp4_it_df.shape[0]
        print(f"Dropped {n_orig - n_final} MP4 instances with duplicate filenames.")

    # drop StudyID column
    if 'StudyID' in mp4_it_df.columns:
        mp4_it_df.drop(columns=['StudyID'], inplace=True)

    # merge the two instance tables
    print(f"Merging main and mp4 instance tables")
    it_df = it_df.merge(
        mp4_it_df,
        on='filename',
        how='left',
        indicator=True,
    ).reset_index(drop=True)

    # construct file type column
    it_df['file_type'] = it_df['filename'].apply(lambda x: x.split('.')[-1])

    # filter out png files
    n_orig = it_df.shape[0]
    it_df = it_df[it_df['file_type'] != 'png']
    n_final = it_df.shape[0]
    print(f"Dropped {n_orig - n_final} PNG instances.")

    # fill NaN values in NumberOfFrames with 1
    it_df['NumberOfFrames'] = it_df['NumberOfFrames'].replace('', np.nan)
    it_df['NumberOfFrames'] = it_df['NumberOfFrames'].fillna(1)

    # convert to int, if NumberOfFrames are saved as str type
    it_df['NumberOfFrames'] = it_df['NumberOfFrames'].apply(lambda x: int(float(x)))
    it_df = it_df.astype({'NumberOfFrames': 'int64'})

    # filter out DICOM videos < min_dicom_frames (DICOMs only)
    # this just filters out still images because threshold is 2
    if 'min_dicom_frames' in info:
        mdf = info['min_dicom_frames']
        n_orig = it_df.shape[0]
        it_df = it_df[np.logical_or(it_df['NumberOfFrames'] >= mdf, it_df['file_type'] == MP4_FILE_TYPE)]
        n_final = it_df.shape[0]
        print(f"Dropped {n_orig - n_final} DICOM still image instances.")

    # replace empty and None tags with "Unknown"
    change = np.logical_or(empty_cells(it_df, 'tag'), it_df['tag'].str.lower() == 'none')
    n_changes = change.sum()
    it_df.loc[change, 'tag'] = 'Unknown'
    print(f"Replaced {n_changes} empty tags with Unknown.")

    # transform (some) columns
    if 'transform_columns' in info['main_IT_file']:
        for col, func in info['main_IT_file']['transform_columns'].items():
            if func == 'relpath':
                # FAMLI2, FAMLI3, and DXA do not list the source relpath, just PID_date_time
                # for these datasets the source relpath must be constructed from PID_date_time
                mrp = partial(make_rel_path, project=project)
                it_df[col] = it_df[col].apply(mrp)
            elif func == 'StudyID':
                # DXA has 1 exam per patient and instance table has no StudyID column
                # StudyID column is created by adding "-1" to PID column
                it_df[func] = it_df[col] + '-1'
            else:
                raise ValueError(f"Invalid transform function: {func}")

    # create a join column for merging exam data
    it_df['join_col'] = create_join_col(it_df, info['main_IT_file']['join_columns'])

    # write the instance data to temporary csv file
    it_path = os.path.join(
        out_dir,
        f"{project}_instance_table_all.csv")
    it_df.to_csv(it_path, index=False)
    print(f"Full instance table written to {it_path}")

    # drop the _merge column
    it_df.drop(columns=['_merge'], inplace=True)

    # copy the instance table to the output dataframe
    out_df = it_df.copy(deep=True)

    # read exam data
    exam_path = os.path.join(
        out_dir,
        info['exam_file']['filename'])
    exam_df = read_spreadsheet(exam_path)
    print(f"Read {exam_df.shape[0]} rows from {info['exam_file']['filename']}.")

    # eliminate all but needed columns
    for col in exam_df.columns:
        if col not in info['exam_file']['columns']:
            exam_df.drop(columns=[col], inplace=True)

    # rename (some) columns
    if 'rename_columns' in info['exam_file']:
        print(f"Renaming columns in exam table")
        exam_df.rename(columns=info['exam_file']['rename_columns'], inplace=True)

    # ManufacturerModelName renames
    if 'manufacturer_model_name_map' in info:
        print(f"Renaming Manufacturer Model names")
        for key, text in info['manufacturer_model_name_map'].items():
            exam_df['ManufacturerModelName'][exam_df['ManufacturerModelName'] == text] = key

    # replace empty Model string with "Unnamed"
    exam_df['ManufacturerModelName'][exam_df['ManufacturerModelName'] == ''] = 'Unnamed'

    # create a join column for merging with instance data
    exam_df['join_col'] = create_join_col(exam_df, info['exam_file']['join_columns'])

    # container for matching join_col = StudyID_Mfr_Model
    exam_join_values = exam_df['join_col'].values
    # container for matching StudyID
    exam_study_ids = exam_df['StudyID'].values
    # container indicating source of exam metadata
    exam_meta_src = exam_df['found_in'].values

    # also drop the join_columns and the join_col
    for col in info['exam_file']['join_columns']:
        exam_df.drop(columns=[col], inplace=True)
    exam_df.drop(columns=['join_col'], inplace=True)

    # add columns for exam data to the output DF
    exam_columns = exam_df.columns.tolist()
    for col in exam_columns:
        out_df[col] = ''

    # add column for source of exam metadata, either 'Match' or 'StudyID'
    out_df['exam_source'] = ''

    # #### merge the instance and exam tables by join_col OR StudyID ####

    # matching join values = StudyID_Mfr_Model in instance table
    it_join_values = out_df['join_col'].values
    # matching StudyIDs in instance table
    it_study_ids = out_df['StudyID'].values

    # drop the join column
    out_df.drop(columns=['join_col'], inplace=True)

    # find unique values of join_col in instance table
    # but also their corresponding StudyIDs
    uniq_it_join_values, first_inds = np.unique(it_join_values, return_index=True)
    uniq_it_study_ids = it_study_ids[first_inds]
    uniq_exam_study_ids = zip(uniq_it_join_values, uniq_it_study_ids)

    # iterate through unique exam values
    for exam, study_id in tqdm(uniq_exam_study_ids,
                               desc="Merging instance and exam tables",
                               total=len(uniq_it_join_values)):
        # get mask for videos in instance table with join_col == exam
        mask = it_join_values == exam

        # find exam in join_values
        exam_match_ind = np.where(exam_join_values == exam)[0]
        if len(exam_match_ind) == 0:
            # no exact match found in exam table
            study_match_ind = np.where(exam_study_ids == study_id)[0]
            if len(study_match_ind) == 0:
                # no match found in exam table by join_col or StudyID
                out_df.loc[mask, 'exam_source'] = 'No_StudyID'
            elif len(study_match_ind) == 1:
                # match found in 1 StudyID
                out_df.loc[mask, 'exam_source'] = 'StudyID'
                out_df.loc[mask, exam_columns] = exam_df.iloc[study_match_ind].values
            elif len(study_match_ind) > 1:
                # multiple matches found in StudyID, pick one by order of priority
                out_df.loc[mask, 'exam_source'] = 'Multiple_StudyIDs'
                # sources for each of the study matches
                sources = exam_meta_src[study_match_ind]
                sorted_match_ind = priority_sort_indices(sources, study_match_ind, match_priority)
                out_df.loc[mask, exam_columns] = exam_df.iloc[sorted_match_ind[0]].values
        elif len(exam_match_ind) == 1:
            # exact match found in join_col
            out_df.loc[mask, 'exam_source'] = 'Exam'
            out_df.loc[mask, exam_columns] = exam_df.iloc[exam_match_ind[0]].values

    # sort metadata instance table on relpath
    out_df.sort_values(by=['relpath'], inplace=True)

    # write the final dataframe to a csv file
    out_path = os.path.join(
        out_dir,
        info['output']['out_file'])
    out_df.to_csv(out_path, index=False)
    print(f"Wrote instance table with metadata to {out_path}")
    print(f"Project {project} with {out_df.shape[0]} instances completed.")


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
