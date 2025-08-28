"""
tablify_sr_v9.py

This script converts the raw structured_reports data from the FAMLIx
datasets into a tabular format. The tabular format is more conducive to
downstream processing and analysis. The table will have one row per exam. An exam
is defined as a unique combination of StudyID and Manufacturer. The columns will
contain the extracted measurements and metadata from the structured reports.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import copy
import yaml
import types
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from statistics import mode
from typing import Tuple, List, Union

from ghlobus.utilities.constants import META_DIR
from ghlobus.utilities.data_utils import read_spreadsheet


def extract_manufacturer_model(mfr_string) -> Tuple[str, str]:
    """
    Extract the manufacturer and model from the manufacturer string.

    Args:
        mfr_string (str): The manufacturer string.

    Returns:
        Tuple[str, str]: A tuple containing the manufacturer and model strings.
    """
    # Check for manufacturer and model
    if 'GE' in mfr_string:
        manufacturer = 'GE'
        try:
            model = mfr_string.split('(')[1].split(')')[0]
        except IndexError:
            model = 'V830'
    elif 'Clarius' in mfr_string:
        manufacturer = 'Clarius'
        model = 'C3HD'
    elif 'Sonosite' in mfr_string:
        manufacturer = 'Sonosite'
        model = 'Turbo'
    else:
        manufacturer = 'Unknown'
        model = 'Unknown'

    return manufacturer, model


def get_tabular_dir(source: str, project: str) -> str:
    """
    Find the tabular data directory for a given project.

    Args:
        source (str): The source directory.
        project (str): The project name.

    Returns:
        List[str, str]: The tabular data directory.
    """
    # Look for Tabular_Data folder in the project folder
    folders = os.listdir(os.path.join(source, project))
    # Go through folders looking for exact form of Tabular_Data
    tabular_dir = None
    for f in folders:
        if 'tabular' in f.lower():
            tabular_dir = f
            break

    return tabular_dir


def get_column_names(column_dict) \
        -> Tuple[str, str, str, str, str, str, str, str, str, str]:
    """
    Get the following columns from the column_dict from yaml file.
        pid
        studyid
        folder
        manufacturer
        tagname
        auxiliary
        tagcontent
        numericvalue
        datevalue
        derivation

    Args:
        column_dict (List[str]): The columns of the structured reports DataFrame.

    Returns:
        Tuple[str * 9]: A tuple containing the columns noted above.
    """
    pid_col = column_dict['pid']
    study_col = column_dict['studyid']
    folder_col = column_dict['folder']
    mfr_col = column_dict['manufacturer']
    der_col = column_dict['derivation']
    tag_col = column_dict['tagname']
    aux_col = column_dict['auxiliary']
    cont_col = column_dict['tagcontent']
    num_col = column_dict['numericvalue']
    date_col = column_dict['datevalue']

    return (pid_col,
            study_col,
            folder_col,
            mfr_col,
            der_col,
            tag_col,
            aux_col,
            cont_col,
            num_col,
            date_col)


def get_value_and_tag(
        row: pd.Series,
        tag_col: str,
        aux_col: str,
        num_col: str,
        date_col: str,
        cont_col: str,
        tag_dict) \
        -> Tuple[str, str]:
    """
    Get the source value and destination tag.

    Args:
        row (pd.Series): The row of the structured reports DataFrame.
        tag_col (str): The tag name column.
        aux_col (str): The auxiliary tag column (some SRs have this).
        num_col (str): The numeric value column.
        date_col (str): The date value column.
        cont_col (str): The tag content column.
        tag_dict (dict): A dictionary with source tags as keys and values
                         consisting of destination tag and data type.

    Returns:
        Tuple[str, str]: Tuple containing source value and destination tag.
    """
    # Get the source tag [and auxiliary tag]
    src_tag = row[tag_col]
    if aux_col:
        aux_id = row[aux_col]
        if aux_id:
            src_tag = src_tag + ', ' + aux_id

    # Determine if the source tag is of interest
    if src_tag not in tag_dict.keys():
        return None, None

    # Get the destination tag and data type (ignore agg_method)
    dst_tag, dtype, _ = tag_dict[src_tag]

    # Get the source value
    if num_col is not None:
        num_val = row[num_col]
    else:
        num_val = None

    # Get the date value
    if date_col is not None:
        date_val = row[date_col]
    else:
        date_val = None

    # Get the content value
    if cont_col is not None:
        cont_val = row[cont_col]
        # check for cm or mm
        if cont_val[-3:] == ' mm':
            divide_by_10 = True
        elif cont_val[-3:] == ' cm':
            divide_by_10 = False
        else:
            divide_by_10 = False
    else:
        cont_val = None
        divide_by_10 = False

    # Process the source value based on the data type
    if dtype == 'float':
        if not num_val:
            src_val = None
        else:
            src_val = float(num_val)
            if divide_by_10:
                src_val /= 10
    elif dtype == 'int':
        if not num_val:
            src_val = None
        else:
            src_val = int(num_val)
    elif dtype == 'date':
        if not date_val:
            src_val = None
        else:
            src_val = row[date_col]
    elif dtype == 'content':
        if not cont_val:
            src_val = None
        else:
            # This is number of fetuses entry, e.g., in SonoSite files
            temp = row[cont_col]
            if 'sole' in temp.lower():
                src_val = 1
            elif 'twin' in temp.lower():
                src_val = 2
            else:
                raise ValueError(f"Unknown content {temp}.")
    else:
        raise ValueError(f"Unknown data type {dtype}.")

    return src_val, dst_tag


def process_nof(values: List) -> int:
    """
    Process the NOF tag values.

    Args:
        values (List): The list of NOF values.

    Returns:
        int: The processed NOF value.
    """
    # Get the number of values
    num_values = len(values)

    # If there are no values, assume singleton
    if num_values == 0:
        value = None
    # If there is one value, return it
    elif num_values == 1:
        value = values[0]
    # if there are an even number, return the largest
    elif num_values % 2 == 0:
        value = max(values)
    # If there is an odd number, return the mode
    else:
        value = mode(values)

    return value


def compute_aggregation(values: List,
                        function: types.FunctionType,
                        dtype: str,
                        ) -> Tuple[Union[float, int], Union[float, None]]:
    """
    Computes an aggregate value and a span from a list of entries.
    Depends on method and data type.

    Args:
        values:        List of values
        function:      Aggregation method.
        dtype:         Data type: 'int', 'date', or 'float'.

    Returns:
        value:         The aggregated value.
        var:           Relative range of values, if applicable.
    """
    if dtype == 'int':
        # For INTEGER data types, round method(values)
        value = int(function(values))
        mean = np.mean(values)
        if mean == 0:
            var = 0
        else:
            var = (max(values) - min(values))/mean
    elif dtype == 'date':
        # For DATE data types, compute function(dates)
        dates = [pd.to_datetime(t, yearfirst=True) for t in values]
        earliest = min(*dates)
        mean_diff = function([t - earliest for t in dates])
        value = pd.Series(pd.to_datetime(earliest + mean_diff)).dt.strftime('%Y-%m-%d').values[0]
        var = 0
    else:
        # For FLOAT data types, compute the function(values)
        value = function(values)
        mean = np.mean(values)
        if mean == 0:
            var = 0
        else:
            var = (max(values) - min(values))/mean

    return value, var


def process_multiples(values: List,
                      derivations: Union[List[str], None],
                      dtype: str,
                      method: str,
                      ) -> Tuple[Union[float, int, None], Union[float, None]]:
    """
    Process multiples in a list of values and derivations.

    Args:
        values (List):               List of values.
        derivations (List or None):  List of derivations.
        dtype (str):                 Data type of the tag
        method (str):                Aggregation method for the tag

    Returns:
        value (float or int):       Processed list of values.
        var (float or int):         Relative span of the values.
    """
    # Get the number of values
    num_values = len(values)

    # set derivations and method to lowercase
    method = method.lower()
    if derivations is not None:
        derivations = [None if x is None else x.lower() for x in derivations]

    # construct method function handles
    if method == 'mean':
        fcn = np.mean
    elif method == 'median':
        fcn = np.median
    elif method == 'min':
        fcn = np.min
    elif method == 'max':
        fcn = np.max
    else:
        raise ValueError(f"Unknown aggregation method {method}.")

    # If there are no values, return None
    if num_values == 0:
        value = None
        var = None
    # If there is only one value, return it
    elif num_values == 1:
        value = values[0]
        var = 0
    # If there are multiple values, return method(values)
    else:
        # Check if there is a derivation directive
        if derivations:
            # There is a derivation directive, check if value already there
            if method in derivations:
                # method is in derivations already
                value = values[derivations.index(method)]
                # remove the derived value and get the span from reduced list
                values.pop(derivations.index(method))
                _, var = compute_aggregation(values, fcn, dtype)
            else:
                # method is not there, must manually compute method(values)
                value, var = compute_aggregation(values, fcn, dtype)
        else:
            # We must manually compute method(values)
            value, var = compute_aggregation(values, fcn, dtype)

    return value, var


def main():
    # Default yaml file
    DEFAULT_YAML = "configs/FAMLI2_sr_GE_v9.yaml"

    # Configure the ArgumentParser
    cli_description = "Convert a structured_reports.csv file to tabular form."

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

    # save frequently used parameters
    raw_dir = info['input']['raw_dir']
    project = info['input']['project']

    # Get the Tabular_data directory
    tabular_dir = get_tabular_dir(raw_dir, project)
    if tabular_dir is None:
        raise ValueError(f"Could not find tabular data directory in {project}.")

    # Read the structured reports file
    sr_path = os.path.join(
        raw_dir,
        project,
        tabular_dir,
        info['input']['sr_file'])
    sr_df = read_spreadsheet(sr_path)
    print(f"Read {sr_df.shape[0]} rows from {info['input']['sr_file']}.")

    # Determine key source column names
    (pid_col,
     study_col,
     folder_col,
     mfr_col,
     der_col,
     tag_col,
     aux_col,
     cont_col,
     num_col,
     date_col,
     ) = get_column_names(info['column_dict'])

    # See if there's a studyid column
    if study_col is None:
        # DXA has only one exam per patient and the CRF has no StudyID column
        # The PID column is copied as StudyID and "-1" is appended for StudyID
        sr_df['StudyID'] = sr_df[pid_col] + '-1'
        study_col = 'StudyID'

    # Determine the number of unique exams
    if folder_col:
        # Get unique exams based on 0-PID, 1-StudyID, 2-Folder, and 3-Manufacturer
        exams_df = sr_df[[pid_col, study_col, folder_col, mfr_col]].drop_duplicates()
    else:
        # Get unique exams based on 0-PID, 1-StudyID, and 2-Manufacturer
        exams_df = sr_df[[pid_col, study_col, mfr_col]].drop_duplicates()
    # Reset the index of the exams DataFrame
    exams_df.reset_index(drop=True, inplace=True)
    # Get the number of unique exams
    n_exams = exams_df.shape[0]
    print(f"Found {n_exams} unique exams.")

    # Construct tags of interest from yaml file
    # For each tag, store the destination column and data type
    tags_of_interest = dict()
    if 'tag_dict' in info and info['tag_dict']:
        for key in info['tag_dict']:
            # dest_tag is column where value will be stored
            dst_tag = info['tag_dict'][key]['dst']
            # data type determines handling of value
            dtype = info['tag_dict'][key]['dtype']
            # agg_method determines how multiple values are aggregated
            agg_method = info['tag_dict'][key].get('method', 'mean')
            for src_tag in info['tag_dict'][key]['src']:
                tags_of_interest[src_tag] = (dst_tag, dtype, agg_method)
    else:
        raise ValueError("No tag_dict found in yaml file.")

    # Create output dictionary
    empty_list = [[] for _ in range(n_exams)]
    # Start with scalar columns
    out_dict = dict(
        PID=exams_df[pid_col].tolist(),
        StudyID=exams_df[study_col].tolist(),
        Manufacturer=copy.deepcopy(empty_list),
        ManufacturerModelName=copy.deepcopy(empty_list),
    )

    # If there's a folder column, add it to the output dictionary
    if folder_col:
        out_dict['Folder'] = exams_df[folder_col].tolist()

    # Add columns for each tag of interest
    # These will, in general, be lists of measurements
    for dst_tag, _, _ in tags_of_interest.values():
        if dst_tag not in out_dict:
            out_dict[dst_tag] = copy.deepcopy(empty_list)
            if der_col and not dst_tag == 'NOF':
                out_dict[dst_tag + '_der'] = copy.deepcopy(empty_list)
                out_dict[dst_tag + '_var'] = copy.deepcopy(empty_list)

    # Configure the progress bar
    tqdm.pandas()

    # Iterate through exams_df (note plural exams) one exam at a time
    for exam_idx, exam_tuple in enumerate(tqdm(
            exams_df.itertuples(index=False),
            desc="Extracting measurements from structured reports",
            total=n_exams)):

        # get mask pertaining to the current exam
        if folder_col:
            # noinspection PyTupleAssignmentBalance
            pid_str, study_id, folder_str, mfr_str = exam_tuple
            exam_mask = np.logical_and.reduce((
                sr_df[pid_col].values == pid_str,
                sr_df[study_col].values == study_id,
                sr_df[folder_col].values == folder_str,
                sr_df[mfr_col].values == mfr_str,
            ))
        else:
            # noinspection PyTupleAssignmentBalance
            pid_str, study_id, mfr_str = exam_tuple
            exam_mask = np.logical_and.reduce((
                sr_df[pid_col].values == pid_str,
                sr_df[study_col].values == study_id,
                sr_df[mfr_col].values == mfr_str,
            ))
        # these are rows structured reports for this exam
        this_exam_sr_df = sr_df.loc[exam_mask]

        # Extract Manufacturer and Model and insert into the output dictionary
        manufacturer, model = extract_manufacturer_model(mfr_str)
        out_dict['Manufacturer'][exam_idx] = manufacturer
        out_dict['ManufacturerModelName'][exam_idx] = model

        # Extract the measurements in this_exam_sr_df one row at a time
        for _, row in this_exam_sr_df.iterrows():
            # Get source tag and destination tag
            src_val, dst_tag = get_value_and_tag(
                row,
                tag_col,
                aux_col,
                num_col,
                date_col,
                cont_col,
                tags_of_interest)
            # If the tag is of interest, record the value for this exam
            if src_val is not None:
                # append value to the list for this column
                out_dict[dst_tag][exam_idx].append(src_val)
                # Each column except for NOF needs its own Derivation column
                if not dst_tag == 'NOF' and der_col is not None:
                    # Form Derivation column name
                    out_der_col = dst_tag + '_der'
                    # Get the Derivation message
                    derivation_value = str(row[der_col])
                    if derivation_value:
                        out_dict[out_der_col][exam_idx].append(derivation_value)
                    else:
                        out_dict[out_der_col][exam_idx].append(None)

    # Convert out_dict to DataFrame
    out_df = pd.DataFrame(out_dict)

    # Write the intermediate DataFrame to a CSV file
    intermediate_file = info['output']['out_file'].replace('.csv', '_intermediate.csv')
    out_path = os.path.join(
        out_dir,
        intermediate_file)
    out_df.to_csv(out_path, index=False)
    print(f"Saved intermediate tabular data to {out_path}.")

    # Get unique tags of interest
    tags, inds = np.unique([t[0] for t in tags_of_interest.values() if not t[0] == 'NOF'], return_index=True)
    dtypes = np.array([t[1] for t in tags_of_interest.values() if not t[0] == 'NOF'])[inds]
    methods = np.array([t[2] for t in tags_of_interest.values() if not t[0] == 'NOF'])[inds]

    # Post-process to handle multiple entries for each exam
    for idx, row in tqdm(
            out_df.iterrows(),
            desc="Adjudicating multiple entries",
            total=n_exams):
        # Process NOF independently
        out_df.iloc[idx]['NOF'] = process_nof(row['NOF'])
        # Process all other tags
        if der_col:
            for tag, dtype, method in zip(tags, dtypes, methods):
                value, var = process_multiples(row[tag], row[tag + '_der'], dtype, method)
                out_df.iloc[idx][tag] = value
                out_df.iloc[idx][tag + '_var'] = var
        else:
            for tag, dtype, method in zip(tags, dtypes, methods):
                value, var = process_multiples(row[tag], None, dtype, method)
                out_df.iloc[idx][tag] = value
                out_df.iloc[idx][tag + '_var'] = var

    # Drop the Derivation columns, if needed
    if der_col:
        for tag in tags:
            out_df.drop(columns=[tag + '_der'], inplace=True)

    # Save final result
    path = os.path.join(
        out_dir,
        info['output']['out_file'])
    out_df.to_csv(path, index=False)
    print(f"Saved tabular SR data to {path}.")


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
