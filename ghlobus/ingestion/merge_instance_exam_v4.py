"""
merge_instance_exam_v4.py

Grabs exam-level metadata from NEJMEvidence (exam) spreadsheet;
Merges with instance-level spreadsheet;
Splits out Training, Tuning, and Test Sets.

Author: Daniel Shea
        Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import yaml
import argparse
import pandas as pd

from tqdm import tqdm
from functools import partial

from ghlobus.utilities.data_utils import read_spreadsheet
from ghlobus.utilities.data_utils import read_spreadsheet_columns


def build_path(row: pd.Series):
    fpath = row['relpath'].split(os.sep)
    fpath[1] = row['exam_dir']
    fpath = os.path.join(*fpath)
    return fpath


def check_path_exists(x, raw_dir):
    return os.path.exists(os.path.join(raw_dir, x))


def main():
    # -------- #  Read input parameters from yaml file,
    #  Step 0  #  Create oft-used variables
    # -------- #

    # Default yaml file
    DEFAULT_YAML = "configs/GA_NEJME_IT_exam_v4.yaml"

    # Configure the ArgumentParser
    cli_description = "Combine instance and exam data to "
    cli_description += "metadata-enhanced instance tables."

    # Add arguments to be read from the command line
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--yaml', default=DEFAULT_YAML, type=str)

    # Extract command line arguments
    args = parser.parse_args()

    # read parameter yaml file
    with open(args.yaml, 'r') as f:
        info = yaml.safe_load(f)

    # Set up initial variables (could be replaced by argparsed variables in future?)
    raw_dir = info['input']['dir']
    crf_dir = info['crf']['dir']
    out_dir = info['output']['dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Source CSVs
    it_file = os.path.join(raw_dir, info['input']['instance']['file'])
    exam_file = os.path.join(raw_dir, info['input']['exam']['file'])
    crf_file = os.path.join(crf_dir, info['crf']['file'])

    # Output CSVs
    merged_it_file = os.path.join(out_dir, info['output']['merged_it_file'])
    missing_it_file = os.path.join(out_dir, info['output']['missing_it_file'])

    # Expected output datasets filenames
    # ('Training', 'Tuning', 'Test Sets')
    expected_datasets_filenames = info['output']['expected_filenames']

    # Expected columns for train, val, test CSVs
    instance_columns = info['input']['instance']['columns']
    crf_columns = info['crf']['columns']
    # noinspection PyUnusedLocal
    crf_lie_map = info['crf']['lie_map']
    exam_columns = info['input']['exam']['columns']
    out_columns = info['output']['columns']
    lie_col = 'lie'

    # Important constant
    instance_threshold = info['input']['instance']['threshold']

    # -------- #  Read input spreadsheets,
    #  Step 1  #  transform to expected format
    # -------- #

    # Use tqdm for progress bars
    tqdm.pandas()

    # Load the instances CSV file to Pandas DataFrame
    instance_df = read_spreadsheet(it_file)
    # Rename columns to match the expected format
    instance_df.rename(columns=instance_columns, inplace=True)

    # Load the Exam CSV file to Pandas DataFrame
    exam_df = read_spreadsheet(exam_file)
    # Rename columns to match the expected format
    exam_df.rename(columns=exam_columns, inplace=True)
    # Drop the 'pid' column as instance table has it already
    exam_df.drop(['pid'], axis=1, inplace=True)

    # Load the CRF to read fetal lie
    crf_df = read_spreadsheet_columns(crf_file,
                                      sheet=info['crf']['sheet'],
                                      columns=list(info['crf']['columns'].keys()))
    # Rename columns to match the expected format
    crf_df.rename(columns=crf_columns, inplace=True)

    # Set empty fetal lie values to the exclude value
    crf_df['us_lie'].replace('', info['crf']['lie_none'], inplace=True)
    # Map the fetal lie values using the provided mapping
    crf_df['us_lie'] = crf_df['us_lie'].astype(int)
    crf_df['us_lie'] = crf_df['us_lie'].map(lambda x: crf_lie_map[x])
    # Rename the fetal lie column to match the expected output
    crf_df.rename(columns={'us_lie': lie_col}, inplace=True)

    # Merge the exam DataFrame with the CRF DataFrame
    exam_df = pd.merge(
        exam_df,
        crf_df,
        how="left",
        on='StudyID',
        sort=True,
        suffixes=("_exam", "_crf"),
        copy=True,
    )

    # -------- #  Create required columns in the Instances DataFrame
    #  Step 2  #
    # -------- #

    # Construct `filename`, `relpath`, `exists`, `file_type` columns
    cpe = partial(check_path_exists, raw_dir=raw_dir)
    instance_df['relpath'] = instance_df.progress_apply(build_path, axis=1)
    instance_df['exists'] = instance_df['relpath'].progress_apply(cpe)
    instance_df['filename'] = instance_df['relpath'].progress_apply(os.path.basename)
    instance_df['file_type'] = 'dcm'

    # -------- #  Remove instances where files are missing
    #  Step 3  #
    # -------- #

    # Get the missing instances; hold on to these for reporting purposes
    missing_instance_df = instance_df.loc[~instance_df['exists']].copy()
    # Drop the 'exists' column as it is no longer needed
    missing_instance_df.drop(['exists'], axis=1, inplace=True)

    # Focus on instances where associated file has been located
    # (i.e. files aren't missing for these instances!)
    instance_df = instance_df.loc[instance_df['exists']]
    # Drop the 'exists' column as it is no longer needed
    instance_df.drop(['exists'], axis=1, inplace=True)

    # -------- #  Removing instances of exams with insufficient
    #  Step 4  #  number of associated instances.
    # -------- #

    # Get the number of instances organized by exam (exam_dir)
    instances_by_exam = instance_df['exam_dir'].value_counts()

    # Look for any exams that have less than `INSTANCE_THRESHOLD` associated instances

    # note: default value will be `INSTANCE_THRESHOLD=2`
    low_data_exams = instances_by_exam[instances_by_exam < instance_threshold].index

    # Convert to set for O(1) membership check
    low_data_exams = set(low_data_exams)

    # Separate out the low data
    # Create a IsLowDataExam selection column
    instance_df['IsLowDataExam'] = instance_df['exam_dir'].apply(lambda x: x in low_data_exams)

    # Separate the LowDataSids
    low_data_exams_df = instance_df.loc[instance_df['IsLowDataExam'] == True]

    # From the remaining data
    instance_df = instance_df.loc[instance_df['IsLowDataExam'] == False].copy()

    # Drop the IsLowDataExam column from both Dataframes
    low_data_exams_df.drop(['IsLowDataExam'], axis=1, inplace=True)
    instance_df.drop(['IsLowDataExam'], axis=1, inplace=True)

    # Add the LowDataExams to the missing_instance_df sheet
    missing_instance_df = pd.concat([missing_instance_df, low_data_exams_df])

    # -------- #  Combine instance table with exam metadata
    #  Step 5  #  Put the order columns in the expected
    # -------- #

    # Merge instance_df and exam_df
    merged_instance_df = pd.merge(
        instance_df,
        exam_df,
        how="left",
        on='StudyID',
        sort=True,
        suffixes=("_instance", "_exam"),
        copy=True,
    )
    # Reorder the columns to match the expected output
    merged_instance_df = merged_instance_df[out_columns]

    # -------- #  Save merged_instance_df and missing_instance_df
    #  Step 6  #  which are the final and missing spreadsheets, respectively.
    # -------- #

    # Output merged instance dataframe
    fpath2 = os.path.join(out_dir, merged_it_file)
    merged_instance_df.to_csv(fpath2, index=False)

    # Output missing instance dataframe
    fpath3 = os.path.join(out_dir, missing_it_file)
    missing_instance_df.to_csv(fpath3, index=False)

    # -------- #  Split data into subsets:
    #  Step 7  #  'Training', 'Tuning', 'Test Sets'
    # -------- #

    # Get the unique set_type values; should be as indicated above
    datasets = list(merged_instance_df['set_type'].unique())

    # Empty result dictionary
    split_data = dict()

    # Split the data
    for dataset in datasets:
        # Get a pd.Series of booleans indicating if the row is part of the dataset
        in_set = merged_instance_df['set_type'] == dataset
        # Save the DataFrame subset of rows that are in the set to the corresponding
        # split_data key.
        split_data[dataset] = merged_instance_df.loc[in_set].copy().reset_index(drop=True)

    # -------- #  Output the split dataframes
    #  Step 8  #
    # -------- #

    # Output Train, Val, and Test dataframes
    for dataset, filename in expected_datasets_filenames.items():
        if dataset not in datasets:
            raise ValueError(f"{dataset} not found in instance CSV datasets")
        fpath4 = os.path.join(out_dir, filename)
        split_data[dataset].to_csv(fpath4, index=False)


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
