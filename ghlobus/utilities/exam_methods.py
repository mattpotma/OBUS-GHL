"""
exam_methods.py

Utility methods for defining exams consisting of blind sweep videos, with different
sweep types and tags. The methods are used to build a DataFrame of exams for inference.

Author: Daniel Shea

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Union


def build_exams_df(
    df: pd.DataFrame,
    columns_defining_exam: List[str],
    known_tag_combos: List[set],
    columns_to_record: List[str],
) -> pd.DataFrame:
    """
    Builds a DataFrame of exams from a given DataFrame.

    Args:
        df: pd.DataFrame                  Input DataFrame.
        columns_defining_exam: List[str]  Columns that define an exam.
        known_tag_combos: List[set]       Known combinations of tags.
        columns_to_record: List[str]      Columns to record.

    Returns:
        pd.DataFrame:                     DataFrame of exams.
    """
    # Set copy of dataframe
    df = df.copy()
    for col in columns_defining_exam:
        df[col].fillna('unknown', inplace=True)
    # Get unique combinations of the given columns
    column_combos = df[list(columns_defining_exam)].copy()
    column_combos.drop_duplicates(inplace=True)
    column_combos.reset_index(drop=True, inplace=True)

    # Exam dictionary, later converted to DataFrame
    exams = defaultdict(list)

    # For each of these 'exams' collect the relevant information
    for _, exam_combo in column_combos.iterrows():
        # Exam row contains only the three columns above, so we'll find
        # dataframe rows that have the 3 given columns
        # first -- build out a list of conditions for selecting relevant rows from dataframe
        conditions = [df[col] == exam_combo[col]
                      for col in columns_defining_exam]
        # Aggregate the row selection conditions
        conditions = sum(conditions) == len(conditions)
        # Now get the exam rows:
        exam_rows = df.loc[conditions].copy()
        # Get data for the dataframe
        exam_dict = parse_exam_row(
            exam_rows, known_tag_combos, columns_to_record)
        # Add the exam_dict to the exams defaultdict (ie filling out the exams dict with data)
        for key in exam_dict.keys():
            exams[key].append(exam_dict[key])

    exams = pd.DataFrame(exams)
    return exams


def parse_exam_row(exam_rows: pd.DataFrame,
                   known_tag_combos: List[set],
                   columns_to_record: List[str],
                   ) \
        -> Dict[str, Union[str, int, List[int], set, None]]:
    """
    Parses an exam row and returns a dictionary of exam data.

    Args:
        exam_rows: pd.DataFrame                            DataFrame of exam rows.
        known_tag_combos: List[set],                       Known combinations of tags.
        columns_to_record: List[str]                       Columns to record.

    Returns:
        Dict[str, Union[str, int, List[int], set, None]]:  Dictionary of exam data.
    """
    exam_dict = {}

    # First, look at the columns to record from the instances dataframe rows ("exam_rows")
    for col in columns_to_record:
        unique_vals = exam_rows[col].unique().tolist()
        if len(unique_vals) == 1:
            exam_dict[col] = unique_vals[0]
        else:
            exam_dict[col] = set(unique_vals)

    # Now compute aggregate numbers
    exam_dict['TotalInstances'] = len(exam_rows.index)
    exam_dict['AllIndices'] = sorted(list(exam_rows.index))
    tagset = set(exam_rows['tag'].tolist())
    exam_dict['AllTagsSet'] = tagset

    # Compare the AllTagsSet against known_tag_combos for a match,
    # note - matches are IN ORDER, which means list of tag sets in
    # known_tag_combos MUST be in order of priority
    exam_tag_combo = None
    if known_tag_combos:
        for tag_combo in known_tag_combos:
            # issuperset() returns True for equal sets, too
            if tagset.issuperset(tag_combo):
                exam_tag_combo = tag_combo
                break
    # Record the matched combo, if any
    exam_dict['ExamTagCombo'] = exam_tag_combo

    return exam_dict


def define_exam(exam_row: pd.Series,
                instances_df: pd.DataFrame,
                max_videos_per_exam: Union[int, None]) \
        -> List[int]:
    """
    Builds exams from each row in the exam DataFrame, by looking at
    instances associated with the exam and selecting up to 6 for 
    analysis. NOTE: THIS IS WHERE 'RANDOMNESS' is introduced to the
    analysis because videos are randomly chosen to construct the exam.

    Args:
        exam_row: pd.Series                 Exam row of the DataFrame.
        instances_df: pd.DataFrame          DataFrame of instances.
        max_videos_per_exam: int or None    Maximum number of videos per exam.

    Returns:
        List[int]:                          List of indices of the selected rows.
    """
    # Get the indices in the instances_df corresponding to instances for this exam
    idcs = exam_row['AllIndices']

    # The ExamTagCombo for the given exam
    exam_tag_combo = exam_row['ExamTagCombo']

    # Access the instances from the instance dataframe
    instance_rows = instances_df.loc[idcs]

    # Option 1: If a known exam_tag_combo can be supplied,
    if exam_tag_combo is not None:
        selected_rows = []
        for tag in exam_tag_combo:
            # Get the rows associated with just this given tag
            tagrows = instance_rows.loc[instance_rows['tag'] == tag]
            # Get one row for each tag
            selected_rows.append(tagrows.sample(n=1))
        # Combine all the rows into a single DataFrame
        selected_rows = pd.concat(selected_rows)
    # Option 2: Specify the exam randomly from the given rows
    else:
        # The number of instances (videos) in an exam
        n_instances = len(instance_rows.index)
        # If a (non-zero) max number of videos is requested
        # And the number of instances exceeds the requested max number of videos
        if (max_videos_per_exam is not None) and \
                (n_instances > max_videos_per_exam):
            # Sample up to `max_videos_per_exam` videos from the instances
            # belonging to this exam.
            selected_rows = instance_rows.sample(
                n=min(max_videos_per_exam, n_instances))
        # Otherwise if the number of instances is less than or equal to the
        # requested max number of videos, or if the max number of videos
        # requested is None,
        else:
            # Return all rows containing instances belonging to the exam.
            selected_rows = instance_rows

    return selected_rows.index.tolist()
