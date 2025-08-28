"""
sweep_utils.py

A collection of useful functions for sweep tag management.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import numpy as np
import pandas as pd

from typing import List, Tuple


# Known combinations of tags
KNOWN_COMBO_TAGS = [
    {'M', 'L1', 'R1', 'C1', 'C2', 'C3'},
    {'M', 'L0', 'R0', 'C1', 'C2', 'C3'},
    {'NM', 'NL', 'NR', 'NC1', 'NC2', 'NC3'},
    {'L15', 'L45', 'M', 'C1', 'R15', 'R45'},
    {'M', 'L1', 'R1', 'L2', 'R2', 'C1'},
    {'M', 'ML', 'MR', 'C1', 'R0', 'L0'},
    {'C1', 'C2', 'C3', 'M', 'ML', 'MR'},
]

# Known combinations of tags
KNOWN_COMBO_TAGS_FP = [
    {'M', 'L1', 'R1'},
    {'M', 'L0', 'R0'},
    {'NM', 'NL', 'NR'},
    {'L15', 'L45', 'M', 'R15', 'R45'},
    {'M', 'L1', 'R1', 'L2', 'R2'},
    {'M', 'ML', 'MR', 'R0', 'L0'},
    {'M', 'ML', 'MR'},
    {'RTA', 'RTB', 'RTC'},
]


COLUMNS_TO_RECORD = {
    'GA': ['StudyID', 'ga_boe', 'ga_anchor'],
    'FP': ['StudyID', 'ga_boe', 'ga_anchor', 'lie'],
    'EFW': ['StudyID', 'log_AC', 'log_FL', 'log_HC', 'log_BPD', 'EFW'],
    'TWIN': ['StudyID', 'GA', 'TWIN'],
}

# Default columns for an exam
DEFAULT_EXAM_COLS = [
    'exam_dir',
]

# Target number of blind sweeps
TARGET_BLIND_SWEEPS = 6

# This data structure allows building of known combos on the fly
# Tags in the current exam are matched against the tags in the
# SWEEP_CONSTRUCTOR in the order given, first Ms, then Ls, then Rs, then Cs.
# The interpretation of each the tuple values for each key are as follows
# 'M': (target#, valid_tags)
# It first looks for target# in each group, matching against valid_tags
# It then proceeds to the next group, and so on until it accumulates
# the required number of tags.

SWEEP_CONSTRUCTOR = {
    # Known combinations of blind sweep tags
    'known_combos': {
        'M': (1, ['M', 'M0', 'M1', 'NM', 'N', 'ASSM', 'CALCM']),
        'L': (1, ['L', 'L0', 'L1', 'L2', 'L3',
                  'L15', 'ML', 'L45',
                  'NL', 'NL0', 'NL1',
                  'ASSL0', 'ASSL1',
                  'CALCL0', 'CALCL1']),
        'R': (1, ['R', 'R0', 'R1', 'R2', 'R3',
                  'R15', 'MR', 'R45',
                  'NR', 'NR0', 'NR1',
                  'ASSR0', 'ASSR1',
                  'CALCR0', 'CALCR1']),
        'C': (3, ['C1', 'C2', 'C3', 'C4', 'C5', 'C6',
                  'NC', 'NC1', 'NC2', 'NC3', 'NC4',
                  'ASSC1', 'ASSC2', 'ASSC3', 'ASSC4',
                  'CALCC1', 'CALCC2', 'CALCC3', 'CALCC4'])},
    # Other blind sweeps, such as assumed blind sweeps and transverse sweeps
    'other_blind_sweeps': {
        'assumed': (6, ['ASSBS']),
        'transverse': (6, ['RTA', 'FA1', 'RTB', 'RTC', 'FA2'])},
    'biometric_sweeps': {
        'biometric': (6, ['HC', 'HC1', 'HC2', 'HC3',
                          'AC', 'AC1', 'AC2', 'AC3',
                          'FL', 'FL1', 'FL2', 'FL3',
                          'BPD', 'BPD1', 'BPD2', 'BPD3',
                          'CRL', 'CRL1', 'CRL2', 'CRL3',
                          'TCD', 'TCD1', 'TCD2', 'TCD3',
                          'HL', 'HL1', 'HL2', 'HL3',
                          'GS', 'YS'])},
    # Completely unknown sweeps (no information)
    'unknown_sweeps': {
        'unknown': (6, ['Unknown'])},
}

# Construct a series of sweep type groups from the SWEEP_CONSTRUCTOR
# VERTICAL_SWEEP_TAGS
# All BLIND_SWEEP_TAGS
# BLIND_AND_BIOMETRIC_TAGS
# BLIND_BIOMETRIC_AND_UNKNOWN_TAGS

# Start the ordered list of SWEEP TAGS with the vertical sweeps
SWEEP_PRIORITY = list()
SWEEP_PRIORITY.extend(SWEEP_CONSTRUCTOR['known_combos']['M'][1])
SWEEP_PRIORITY.extend(
    np.array(list(zip(
        SWEEP_CONSTRUCTOR['known_combos']['L'][1],
        SWEEP_CONSTRUCTOR['known_combos']['R'][1])))
    .flatten()
    .tolist())

# Vertical sweep tags (for FP training and inference)
VERTICAL_SWEEP_TAGS = SWEEP_PRIORITY.copy()

# Vertical plus transverse sweep tags (for FP training and inference)
VERTICAL_PLUS_TRANSVERSE = SWEEP_PRIORITY.copy()
VERTICAL_PLUS_TRANSVERSE.extend(SWEEP_CONSTRUCTOR['other_blind_sweeps']['transverse'][1])

# Continue adding the horizontal blind sweep tags
SWEEP_PRIORITY.extend(SWEEP_CONSTRUCTOR['known_combos']['C'][1])
# Continue adding the assumed blind sweep tags
SWEEP_PRIORITY.extend(SWEEP_CONSTRUCTOR['other_blind_sweeps']['assumed'][1])
# Continue adding the transverse blind sweep tags
SWEEP_PRIORITY.extend(SWEEP_CONSTRUCTOR['other_blind_sweeps']['transverse'][1])

# Keep a record of what are actual blind_sweeps
BLIND_SWEEP_TAGS = SWEEP_PRIORITY.copy()

# Blind sweeps and biometric sweeps
SWEEP_PRIORITY.extend(SWEEP_CONSTRUCTOR['biometric_sweeps']['biometric'][1])
BLIND_AND_BIOMETRIC_TAGS = SWEEP_PRIORITY.copy()

# Blind, Biometric, and Unknown sweeps
SWEEP_PRIORITY.extend(SWEEP_CONSTRUCTOR['unknown_sweeps']['unknown'][1])
BLIND_BIOMETRIC_AND_UNKNOWN_TAGS = SWEEP_PRIORITY.copy()


def relative_sort(arr1: List[str], arr2: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort one array based on an order defined by another array.

    Parameters
    ----------
        arr1: List[str]      Array to sort.
        arr2: List[str]      Array according to which to sort.

    Returns
    -------
        Tuple of :
                arr1_sorted: List[str]    Sorted array 1.
                order: List[int]          Order of indices that arr1 was sorted by.
    """
    arr2_numeric = np.arange(len(arr2), dtype=int)
    arr1_numeric = np.array(
        [arr2_numeric[arr2.index(t)] if t in arr2 else 9999 for t in arr1])
    order = np.argsort(arr1_numeric)
    arr1_sorted = np.array([arr1[i] for i in order])
    return arr1_sorted, order


def get_known_combo_indices(tags: List[str],
                            allow_biometric: bool = False,
                            strict_known_combos: bool = False,
                            random: bool = False) -> List[int]:
    """
    Takes the tags of the current exam and returns indices corresponding to
        a constructed known_combo.
    Parameters
    ----------
        tags: List[str]            List of tags in current exam
        allow_biometric: bool      Whether to use biometric sweeps when there aren't
                                       enough blind sweeps
        strict_known_combos: bool  Whether to use strict definition of known_combos
        random: bool               Whether to take random sample or fixed known_combo

    Returns
    -------
        List[int]
    """
    # Put the tags in the priority order
    input_tags = tags.copy()
    ordered_tags, order = relative_sort(input_tags, SWEEP_PRIORITY)

    # Container for selected indices
    indices = list()

    # Look for tag matches in the order provided by SWEEP_CONSTRUCTOR
    n_matches = 0
    for group, constructor in SWEEP_CONSTRUCTOR.items():
        # Check if we are enforcing strict definition of known_combos
        if group == 'other_blind_sweeps' and strict_known_combos:
            break
        # Check if we are allowing biometric sweeps
        if group == 'biometric_sweeps' and not allow_biometric:
            break
        # Initialize container for extra indices beyond the target, in case they are needed
        extra_indices = list()
        for tag_type, (target_count, valid_tags) in constructor.items():
            match = np.isin(ordered_tags, valid_tags)
            if match.any():
                # there's at least one matching tag
                matching_inds = np.where(match)[0]
                # ideal number to take from this group
                num_to_take = min(
                    min(TARGET_BLIND_SWEEPS - n_matches, target_count), len(matching_inds))
                # Take random match or highest priority match?
                if random:
                    keep_inds = np.random.choice(
                        matching_inds, num_to_take, replace=False)
                    indices.extend(keep_inds)
                else:
                    indices.extend(matching_inds[:num_to_take])
                # see if there are more available, in case further groups don't find matches
                if num_to_take < len(matching_inds):
                    # there ARE more available, so let's stash them
                    if random:
                        # if we took random indices, we need to find remaining ones
                        #     and if so, keep_inds variable is guaranteed to exist
                        # noinspection PyUnboundLocalVariable
                        remaining_inds = list(
                            set(matching_inds) - set(keep_inds))
                        extra_indices.extend(remaining_inds)
                    else:
                        extra_indices.extend(matching_inds[num_to_take:])
                # update the number of matches
                n_matches = len(indices)
            # Check if we have enough matches
            if n_matches >= TARGET_BLIND_SWEEPS:
                # we have enough matches, so break out of the loop
                break
        # Check if we are enforcing strict definition of known_combos
        if group == 'known_combos' and strict_known_combos:
            break
        # Check if we have enough sweeps; if not supplement with stashed tags
        if n_matches < TARGET_BLIND_SWEEPS:
            # We don't have enough sweeps, so let's add some more
            n_more = min(TARGET_BLIND_SWEEPS - n_matches, len(extra_indices))
            if random:
                # Choose randomly from extra indices
                more_inds = np.random.choice(
                    extra_indices, n_more, replace=False)
                indices.extend(more_inds)
            else:
                # Choose highest priority from extra indices
                indices.extend(extra_indices[:n_more])
            # update the number of matches
            n_matches = len(indices)
    # Must convert indices back to original array's order
    indices = list(order[indices])

    # sort before returning so videos stay in order
    return sorted(indices)


def filter_by_known_combos(df: pd.DataFrame,
                           allow_biometric: bool = False,
                           strict_known_combos: bool = False) -> pd.DataFrame:
    """
    Filter the dataframe by KNOWN_COMBO_TAGS at the exam level
    Parameters
    ----------
        df: pd.DataFrame           Dataframe tabulating the data by patient, exam, and video
        allow_biometric: bool      Whether to use biometric sweeps when insufficient KNOWN_COMBO_TAGS
        strict_known_combos: bool  Whether to use strict definition of KNOWN_COMBO_TAGS

    Returns
    -------
        df2: pd.DataFrame          Dataframe where each exam is filtered by KNOWN_COMBO_TAGS
    """
    # Get the exams
    exams = list(set(df['exam_dir'].tolist()))
    # Reset index of dataframe
    new_df = df.copy()
    # Keep track of indices to keep
    keep_indices = list()
    # Go through each exam in turn
    for exam in exams:
        # Get mask for this exam
        this_exam = new_df['exam_dir'] == exam
        # Get indices for this exam
        this_exam_inds = new_df.index[this_exam].values
        # Get the tags for this exam
        exam_tags = df.iloc[this_exam_inds]['tag'].values
        # Filter by KNOWN_COMBO_TAGS
        keep_inds = get_known_combo_indices(exam_tags,
                                            allow_biometric=allow_biometric,
                                            strict_known_combos=strict_known_combos,
                                            random=False)
        # Accumulate keep indices
        keep_indices.extend(this_exam_inds[keep_inds])

    # Return the dataframe filtered by keep_indices
    return new_df.iloc[keep_indices]


def get_tag_selection(tag_sel_dict: dict) -> List[str]:
    """
    Get the tags for the tag selection dictionary that is read from a yaml file.

    Parameters
    ----------
        tag_sel_dict: dict    Dictionary containing the tag specifications.

    Returns
    -------
        List[str]:            Tag selections intended by the user.
    """
    # Default case is only blind sweeps
    tags = BLIND_SWEEP_TAGS
    # Check if Biometric sweeps are also included
    if 'include_biometric' not in tag_sel_dict:
        # Done; just Blind sweeps
        pass
    elif tag_sel_dict['include_biometric']:
        # Include Blind and Biometric sweeps
        tags = BLIND_AND_BIOMETRIC_TAGS
        # Check if Unknown sweeps are also included
        if 'include_unknown' not in tag_sel_dict:
            # Done; just blind and biometric sweeps
            pass
        elif tag_sel_dict['include_unknown']:
            # Include Blind, Biometric, and Unknown sweeps
            tags = BLIND_BIOMETRIC_AND_UNKNOWN_TAGS
    else:
        # Include only Blind sweeps
        pass
    return tags
