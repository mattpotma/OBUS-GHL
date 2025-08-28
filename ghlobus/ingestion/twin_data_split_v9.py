"""
twin_data_split_v9.py

This module splits data into train, validation, and test sets
according to specified fractions of the total data.
or balanced samples using a given fraction of the data,
and a balance weight between 0 and 1.

This module permits filtering:
 - filtering on gestational age (GA).
   Set ga_threshold to 0 to disable.
 - filtering on sweep tags.
   Set tags to null (in yaml file) to disable.

The balance value can be continuously adjusted between 0 and 1:
 balance = 0 pure stratified, i.e., matching source distribution
 balance = 1 pure balance, i.e., all distribution bins are equal
 balance = 0.5 midway between stratified and completely balanced

For continuous features, use strategy = 'combo' so that final counts
match the target count.

The strategy for balancing when balance = 1 can be set:
 strategy = 'over' only use oversampling
 strategy = 'under' only use undersampling
 strategy = 'combo' use over- and under-sampling to achieve target count

For the case of no balancing or stratification:
 strategy = 'all' use all samples without balancing or stratification

Author: Courosh Mehanian

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
import matplotlib.pyplot as plt

from functools import partial

from ghlobus.utilities.constants import META_DIR
from ghlobus.utilities.plot_utils import Logger
from ghlobus.utilities.data_utils import read_spreadsheet_columns
from ghlobus.utilities.data_utils import construct_outpath
from ghlobus.utilities.sample_utils import construct_bins
from ghlobus.utilities.sample_utils import balanced_sample
from ghlobus.utilities.sample_utils import bin_discrete_feature
from ghlobus.utilities.biometry_utils import fill_missing_ga_values
from ghlobus.utilities.sweep_utils import get_tag_selection


def search_first_1d(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    """
    Find the first indices of the elements of array_b in array_a.

    Parameters:
        array_a: np.ndarray   Array to search in.
        array_b: np.ndarray   Array to search for.

    Returns:
        np.ndarray:           First indices of array_b in array_a.
    """
    return np.array([np.where(array_a == x)[0][0] for x in array_b])


def search_all_1d(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    """
    Find all indices of the elements of array_b in array_a.

    Parameters:
        array_a: np.ndarray   Array to search in.
        array_b: np.ndarray   Array to search for.

    Returns:
        np.ndarray:           Indices of array_b in array_a.
    """
    temp_list = [np.where(array_a == x)[0] for x in array_b]
    return np.concatenate(temp_list).ravel()


def main():
    # default yaml file
    DEFAULT_YAML = "configs/twin_5fold_split_v9.yaml"

    # configure the ArgumentParser
    cli_description = 'Split data into train, val, and test sets at given fractions.\n'
    cli_description += 'Or create k-fold cross-validation splits along with a test set.\n'
    cli_description += 'Specify the feature of interest to do stratified split on.\n'
    cli_description += 'Optionally, splits can be balanced on a separate feature, e.g. GA.\n'

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
                                info['output']['distribution'])
    else:
        out_path = os.path.join(info['output']['root_dir'],
                                info['output']['folder'])
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    # read the splits dictionary
    splits = info['splits']

    # Create test split folder
    name = f"test"
    if 'instance_balance' in splits[name] and splits[name]['instance_balance']:
        splits[name]['folder'] = os.path.join(out_path, f"test")
        if not os.path.exists(splits[name]['folder']):
            os.makedirs(splits[name]['folder'], exist_ok=True)
    else:
        splits[name]['dd_folder'] = os.path.join(out_path, f"test_dd")
        if not os.path.exists(splits[name]['dd_folder']):
            os.makedirs(splits[name]['dd_folder'], exist_ok=True)

    # For k-fold cross-validation create k-duplicate splits of val and train
    if 'crossval' in info:
        k_folds = info['crossval']
        if k_folds and k_folds > 1:
            # k-duplicate 'val' split
            for k in range(k_folds):
                name = f"val_{k:01}"
                splits[name] = splits['val'].copy()
                splits[name]['index'] = 1 + 2*k
                splits[name]['folder'] = os.path.join(out_path, f"{k:01}")
                if not os.path.exists(splits[name]['folder']):
                    os.makedirs(splits[name]['folder'], exist_ok=True)
                splits[name]['dd_folder'] = os.path.join(out_path, f"{k:01}_dd")
                if not os.path.exists(splits[name]['dd_folder']):
                    os.makedirs(splits[name]['dd_folder'], exist_ok=True)
            del splits['val']
            # k-duplicate 'train' split
            for k in range(k_folds):
                name = f"train_{k:01}"
                splits[name] = splits['train'].copy()
                splits[name]['index'] = 1 + 2*k + 1
                splits[name]['folder'] = os.path.join(out_path, f"{k:01}")
                if not os.path.exists(splits[name]['folder']):
                    os.makedirs(splits[name]['folder'], exist_ok=True)
                splits[name]['dd_folder'] = os.path.join(out_path, f"{k:01}_dd")
                if not os.path.exists(splits[name]['dd_folder']):
                    os.makedirs(splits[name]['dd_folder'], exist_ok=True)
            del splits['train']
        else:
            # no k-fold cross-validation
            k_folds = 1
    else:
        # no k-fold cross-validation
        k_folds = 1

    # get target feature name
    target = info['target']['name']

    # save console output to file
    sys.stdout = Logger(os.path.join(out_path, f"console.txt"))

    # Read input files
    in_files = list()
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
        in_files.append(this_df)

    # Read the twin split file
    twin_splits_df = read_spreadsheet_columns(
        os.path.join(info['twin_splits']['except_dir'],
                     info['twin_splits']['file']),
        sheet=None,
        rows=None,
        columns=info['twin_splits']['columns'])
    # Record the twin_split_column
    twin_split_cols = [x for x in info['twin_splits']['columns'] if 'split' in x.lower()]
    if len(twin_split_cols) != 1:
        raise ValueError(f"Expected exactly one column with 'split' in its name, "
                         f"but found {len(twin_split_cols)}: {twin_split_cols}")
    twin_split_col = twin_split_cols[0]

    # stack all the data frames
    it_df = pd.concat(in_files, ignore_index=True)
    # sort by StudyID
    it_df.sort_values(by=['exam_dir', 'filename'], inplace=True)
    # reset the index
    it_df.reset_index(drop=True, inplace=True)

    # branch on target feature type
    if info['target']['type'] == 'continuous':
        raise ValueError("Continuous features not yet supported for this script.")
    elif info['target']['type'] == 'discrete':
        # see if binning specified
        if info['target']['nbins']:
            # bin target feature according to yaml target.nbins
            y = bin_discrete_feature(it_df[info['target']['name']].values.astype(int),
                                     info['target']['nbins'])
        else:
            # no binning needed
            y = it_df[info['target']['name']].values.astype(int)
    else:
        raise ValueError(f"Feature type {info['target']['type']} not supported.")

    # filter rows with negative y values (unwanted feature values)
    if np.any(y < 0):
        print(f"{np.sum(y < 0)} rows have been filtered.")
        it_df = it_df.loc[y >= 0]
        it_df.reset_index(drop=True, inplace=True)
        y = y[y >= 0]

    # get tag values
    tag = it_df['tag'].values
    # replace empty string with 'Unknown'
    tag[tag == ''] = 'Unknown'

    # splits must be made at the patient level
    # DEFINITIONS:
    #     y:                     instance target feature values, e.g., 0 - n_classes-1
    #     tag:                   instance sweep tag values, e.g., 'M', 'R', 'L', etc.
    #     pid:                   patient ID, e.g., FAM-025-001, FAM-025-002, ...
    #     sid:                   study ID, e.g., FAM-025-001-1, FAM-025-001-2, ...
    #     ipid:                  integer patient ID, e.g., 0 - n_patients-1
    #     isid:                  integer study IDs, e.g., 0 - n_studies-1
    #     y_vals:                unique target feature values, e.g., [0, 1]
    #     y_counts:              counts of each target feature value, e.g., [n0, n1]
    #     k_folds:               number of cross-validation folds, e.g., 5
    #     n_classes:             number of unique target feature values, e.g., 2
    #     n_patients:            number of unique patients, e.g., 100
    #     n_instances:           number of instances, e.g., 1000
    #     splits:                split dictionary, e.g., {'train': {'fraction': 0.75}, ...}
    #     index_dict:            index dictionary for each target feature value and split
    #     feature:               stratification feature, e.g., GA
    #     feature_binned:        stratification feature binned, e.g., 0 - n_bins-1
    #     this_y:                current target feature value, e.g., 0
    #     this_y_indices:        absolute indices for this_y, e.g., 0 - n_instances-1
    #     this_y_count:          instance count for this_y, e.g., n0
    #     this_y_mask:           instance mask for this_y
    #     this_y_ipids:          unique ipids for this_y
    #     this_y_pt_count:       patient count for this_y
    #     this_y_pt_first_index: first (absolute) patient index for this_y
    # this_y_split_pt_first_index: first (absolute) patient index for this_y, this_split
    #     this_y_isids:          unique isids for this_y
    #     this_y_ex_count:       exam count for this_y
    #     this_y_ex_first_index: first (absolute) exam index for this_y
    # this_y_split_ex_first_index: first (absolute) exam index for this_y, this_split
    #     rel_ind:               relative index into ipids for this_y, this_split
    #     this_y_split_ipids:    ipids for this_y, this_split
    #     this_y_split_pt_count: patient count for this_y, this_split
    #     this_y_split_ex_count: exam count for this_y, this_split
    #     this_y_split_it_count: instance count for this_y, this_split
    #     selected_ipids:        ipids selected so far for this_y
    #     available_ipids:       ipids available to choose from, after selected ones removed
    #     this_y_split_ipids:    ipids for this_y, this_split
    #     this_y_split_indices:  absolute indices for this_y, this_split

    # get unique patient IDs and assign integer indices to them
    unique_pids, ipids = np.unique(it_df['PID'].values, return_inverse=True)
    unique_sids, isids = np.unique(it_df['exam_dir'].values, return_inverse=True)

    # print some information
    print(f"Total number of patients: {len(unique_pids)},"
          f"                exams: {len(unique_sids)},"
          f"                instances: {it_df.shape[0]}")

    # get unique target feature values and their counts
    y_vals, y_counts = np.unique(y, return_counts=True)

    # container for index dictionary
    index_dict = {y_val: {split: None for split in splits} for y_val in y_vals}

    # container for split allocation
    alloc_dict = {y_val: {'PID': None, 'GA': None} for y_val in y_vals}

    # deal with missing values in stratification feature
    feature, source = fill_missing_ga_values(it_df)
    it_df['GA'] = feature
    it_df['source'] = source
    # reorder columns so `source` is next to `GA`
    cols = it_df.columns.tolist()
    ga_idx = cols.index('GA')
    cols = cols[:ga_idx + 1] + [cols[-1]] + cols[ga_idx + 1:-1]
    it_df = it_df[cols]

    # digitize the stratification feature
    ga_bins = construct_bins(feature, info['processing']['nbins'], balance=1)
    feature_binned = np.digitize(feature, ga_bins) - 1

    # print information exam-level balancing
    print(f"############## EXAM-LEVEL BALANCING ##############")

    # iterate through the unique y values
    fig, axs = plt.subplots(4, 1 + k_folds, figsize=(4 * (k_folds + 1), 16))
    for this_y, this_y_count in zip(y_vals, y_counts):
        # count number of patients in this class, y == this_y
        this_y_mask = np.array(y == this_y)
        # grab the absolute instance indices for this class
        this_y_indices = np.where(this_y_mask)[0]

        # unique ipids, isids for this_y class
        this_y_ipids = np.unique(ipids[this_y_mask])
        this_y_pt_count = len(this_y_ipids)
        this_y_isids = np.unique(isids[this_y_mask])
        this_y_ex_count = len(this_y_isids)
        print(f"{target}={this_y}: "
              f"  patients: {this_y_pt_count}, "
              f"  exams: {this_y_ex_count}, "
              f"  instances: {len(this_y_indices)}.")

        # get the actual PIDs for this_y
        this_y_pids = unique_pids[this_y_ipids]

        # create dictionary of arrays for split allocations
        for split in splits:
            alloc_dict[this_y][split] = \
                np.full(shape=len(this_y_ipids), fill_value='', dtype=object)

        # create ipid->isid mapping, and twin tag pareto
        ipid_to_isid = dict()
        for ipid_idx, this_ipid in enumerate(this_y_ipids):
            # get the indices for this ipid
            inds = np.where(ipids == this_ipid)[0]
            # populate dictionary
            ipid_to_isid[this_ipid] = np.unique(isids[inds])

        # first indices (absolute) of these ipids in the instance table
        this_y_pt_first_index = search_first_1d(ipids, this_y_ipids)
        # first indices (absolute) of these isids in the instance table
        this_y_ex_first_index = search_first_1d(isids, this_y_isids)
        # ipids selected so far
        selected_ipids = np.empty(0, dtype=int)

        # iterate through the splits within this_y value
        for split, subset in sorted(splits.items(), key=lambda x: x[1]['index']):
            # Get the target number of patients
            this_y_split_pt_count = subset['counts'][this_y]

            # determine unavailable ipids for train split
            # for test and val splits, it's just the accumulated selected ipids
            # but for train split, it's the test ipids and the previously selected val ipids
            if 'train' in split:
                # noinspection PyUnboundLocalVariable
                unavailable_ipids = np.append(test_ipids, current_fold_val_ipids)
            else:
                unavailable_ipids = selected_ipids.copy()

            # Determine the available ipids for this_y, this_split
            if this_y == 1:
                # For twins use the twin split file to get the available ipids
                # Compute the name of the split to search for
                if 'test' in split:
                    # if it's the test split, split_search = Test
                    split_search = ['Test']
                elif 'val' in split:
                    # if it's a val split, split_search = FoldX
                    split_search = [f"Fold{int(split.split('_')[1]):1}"]
                else:
                    # if it's a train split, split_search = Fold0 ... Fold4 \ FoldX
                    fold_index = int(split.split('_')[1])
                    split_search = [f"Fold{x:1}" for x in range(k_folds) if x != fold_index]
                # get mask of val_xx Split in the twin_splits_df
                split_mask = np.isin(twin_splits_df[twin_split_col], split_search)
                split_pids = np.unique(twin_splits_df.loc[split_mask, 'PID'].values)
                available_ipids = np.where(np.isin(unique_pids, split_pids))[0]
            else:
                # For singletons, the available ipids are this_y's ipids minus the unavailable ones
                available_ipids = np.setdiff1d(this_y_ipids, unavailable_ipids)

            # mask into first patient indices
            pt_first_index_mask = np.isin(this_y_ipids, available_ipids)
            # compute current first patient indices
            this_y_split_pt_first_index = this_y_pt_first_index[pt_first_index_mask]

            # create stratified sample for this split
            n_patients = len(this_y_split_pt_first_index)

            # to balance or not to balance, that is the question
            if subset['exam_balance'][this_y]:
                # yes, sample by balancing on feature
                # take this fraction of the available patients randomly
                fraction = float(this_y_split_pt_count) / n_patients
                # indices are relative to the table of ipids for this_y
                rel_inds = balanced_sample(feature_binned[this_y_split_pt_first_index],
                                           ftr=None,
                                           fraction=fraction,
                                           balance=info['exam']['balance'],
                                           strategy=info['exam']['strategy'])
            elif this_y == 0:
                # no, take random sample equal to fraction
                base = list(range(n_patients))
                # take this fraction of the available patients randomly
                fraction = float(this_y_split_pt_count) / n_patients
                n_target = min(n_patients, round(fraction * n_patients))
                rel_inds = random.sample(base, k=n_target)
            else:
                # For twins take all of the available ipids
                rel_inds = list(range(n_patients))

            # these are the ipids for this_y, this_split
            this_y_split_ipids = available_ipids[rel_inds]
            this_y_split_ipids.sort()
            # remember selected ipids for holdout test set and current fold val split
            # will be used to compute the available ipids for current fold train split
            if 'test' in split:
                test_ipids = this_y_split_ipids.copy()
            if 'val' in split:
                current_fold_val_ipids = this_y_split_ipids.copy()
            # accumulate selected ipids for test and val splits
            if 'test' in split or 'val' in split:
                selected_ipids = np.append(selected_ipids, this_y_split_ipids)
            # populate the allocation table with these selections
            this_y_split_mask = np.isin(this_y_ipids, this_y_split_ipids)
            alloc_dict[this_y][split][this_y_split_mask] = split
            # number of patients for this split
            this_y_split_pt_count = len(this_y_split_ipids)
            # need to get their absolute instance indices
            this_y_split_indices = search_all_1d(ipids, this_y_split_ipids)
            # are we filtering on sweep tags for this_y and this_split?
            if subset['tags'][this_y]:
                # yes, we are filtering on sweep tags
                # construct tags for this_y, this_split to filter on
                tags = get_tag_selection(subset['tags'][this_y])
                n_orig = len(this_y_split_indices)
                this_y_split_indices = this_y_split_indices[
                    np.isin(tag[this_y_split_indices], tags)]
                n_final = len(this_y_split_indices)
                print(f"  For {split.upper()} split, filtered out {n_orig - n_final} invalid sweep instances")
            # number of exams for this split
            this_y_split_ex_count = len(np.unique(isids[this_y_split_indices]))
            # number of instances for this split
            this_y_split_it_count = len(this_y_split_indices)
            # store indices
            index_dict[this_y][split] = this_y_split_indices
            print(f"  {split.upper()} split: "
                  f"  patients: {this_y_split_pt_count}, "
                  f"  exams: {this_y_split_ex_count}, "
                  f"  instances: {this_y_split_it_count}.")

        # build the exam-based GA strings based on ipid_to_isid
        ga_str = np.full(shape=len(this_y_ipids), fill_value='', dtype=object)
        ga_vals = {split: list() for split in splits}
        for ipid_idx, this_ipid in enumerate(this_y_ipids):
            # get the isids for this ipid
            # noinspection PyUnboundLocalVariable
            this_isids = ipid_to_isid[this_ipid]
            # get the GA values for this ipid
            inds = this_y_ex_first_index[np.isin(this_y_isids, this_isids)]
            this_ipid_gas = [x for x in feature[inds]]
            for split in splits:
                if alloc_dict[this_y][split][ipid_idx]:
                    ga_vals[split].extend(this_ipid_gas)
            ga_str[ipid_idx] = ', '.join(str(round(x, 1)) for x in this_ipid_gas)
        # add to the allocation dictionary
        alloc_dict[this_y]['PID'] = this_y_pids
        # noinspection PyTypedDict
        alloc_dict[this_y]['GA'] = ga_str

        # histogram of exam-level GA values for this y_val, split
        for split, subset in sorted(splits.items(), key=lambda x: x[1]['index']):
            split_idx = subset['index']
            if split_idx == 0:
                r = this_y
                c = 0
            elif split_idx % 2 == 1:
                r = this_y
                c = split_idx // 2 + 1
            else:
                r = 2 + this_y
                c = split_idx // 2
            axs[r, c].hist(ga_vals[split], bins=ga_bins)
            axs[r, c].set_title(f"{split} split_{target}_{this_y:01}")
            if r == 3:
                axs[r, c].set_xlabel("GA (days)")

        # create the dataframe and save it
        alloc_df = pd.DataFrame(alloc_dict[this_y])
        alloc_path = os.path.join(out_path, f"split_tag_pareto_{target}_{this_y:01}.csv")
        alloc_df.to_csv(alloc_path, index=False)

    # finish and save exam-level histograms
    for r in range(2, 4):
        axs[r, 0].remove()
    plt.suptitle(f"Exam-level GA histograms")
    plt.show()
    plt.savefig(os.path.join(out_path, f"exam_level_ga_hist.png"))
    plt.close()

    # done with y, delete it
    del y

    # print information instance-level balancing
    print(f"############ INSTANCE-LEVEL BALANCING ############")

    # ready for balancing across target value
    fig, axs = plt.subplots(6, 1 + k_folds, figsize=(4 * (k_folds + 1), 24))
    for split, subset in sorted(splits.items(), key=lambda x: x[1]['index']):
        split_idx = subset['index']
        # create a new dataframe for this split
        split_df = pd.DataFrame(columns=it_df.columns)
        split_df.reset_index(drop=True, inplace=True)
        # create container for y, ipid, isid values
        y = np.empty(0, dtype=int)
        ga = np.empty(0, dtype=float)
        ga_binned = np.empty(0, dtype=int)
        ipid = np.empty(0, dtype=int)
        isid = np.empty(0, dtype=int)
        # print header about this split
        print(f"{split.upper()} SPLIT")

        # accumulate across unique y values
        for this_y in y_vals:
            # grab the indices for this_y, this_split
            this_y_split_indices = index_dict[this_y][split]
            # append the rows to the split dataframe
            split_df = pd.concat((split_df, it_df.loc[this_y_split_indices]), axis=0)
            # construct the y, ga, ga_binned, ipid, isid arrays for this split
            y = np.append(y, np.full(len(this_y_split_indices), this_y))
            ga = np.append(ga, feature[this_y_split_indices])
            ga_binned = np.append(ga_binned, feature_binned[this_y_split_indices])
            ipid = np.append(ipid, ipids[this_y_split_indices])
            isid = np.append(isid, isids[this_y_split_indices])

        # now balance the split according to directives
        if subset['instance_balance']:
            # do balance vs stratify, choose fraction = 1
            keep_inds = balanced_sample(
                y,
                ftr=ga_binned,
                fraction=1,
                balance=info['instance']['balance'],
                strategy=info['instance']['strategy'])
        else:
            # use all samples
            keep_inds = np.arange(split_df.shape[0])

        # First save the un-duplicated (unbalanced dataframe)
        # write distribution csv
        if 'dd_folder' in subset:
            print(f"  Writing unbalanced {subset['file']} to {subset['dd_folder']}")
            split_df.to_csv(os.path.join(subset['dd_folder'],
                                         subset['file']), index=False)

        # with keep indices in hand, construct the balanced dataframe
        split_df = split_df.iloc[keep_inds]
        split_df.reset_index(drop=True, inplace=True)
        y = y[keep_inds]
        ga = ga[keep_inds]
        ipid = ipid[keep_inds]
        isid = isid[keep_inds]

        # histogram for this split (all y-values)
        if split_idx == 0:
            r = 2
            c = 0
        elif split_idx % 2 == 1:
            r = 2
            c = split_idx // 2 + 1
        else:
            r = 5
            c = split_idx // 2
        axs[r, c].hist(ga, bins=info['processing']['hbins'])
        axs[r, c].set_title(f"{split} split")
        if r == 5:
            axs[r, c].set_xlabel("GA (days)")

        # compute statistics for this split and report
        for this_y in y_vals:
            this_y_split_mask = np.in1d(y, this_y)
            this_y_split_ipids = np.unique(ipid[this_y_split_mask])
            this_y_split_isids = np.unique(isid[this_y_split_mask])
            this_y_split_ga = ga[this_y_split_mask]
            print(f"  {target}={this_y}: "
                  f"  patients: {len(this_y_split_ipids)}, "
                  f"  exams: {len(this_y_split_isids)}, "
                  f"  instances: {this_y_split_mask.sum()}.")
            # histogram by this_y and this split
            if split_idx == 0:
                r = this_y
                c = 0
            elif split_idx % 2 == 1:
                r = this_y
                c = split_idx // 2 + 1
            else:
                r = 3 + this_y
                c = split_idx // 2
            axs[r, c].hist(this_y_split_ga, bins=info['processing']['hbins'])
            axs[r, c].set_title(f"{split} split for {target}={this_y}")

        # write distribution csv
        if 'folder' in subset:
            # write the split dataframe to a csv file
            print(f"  Writing balanced {subset['file']} to {subset['folder']}")
            split_df.to_csv(os.path.join(subset['folder'],
                                         subset['file']), index=False)

    # close off histograms and write file
    for r in range(3, 6):
        axs[r, 0].remove()
    plt.suptitle(f"Video-level GA histograms")
    plt.show()
    plt.savefig(os.path.join(out_path, f"video_level_ga_hist.png"))
    plt.close()

    # close stdout
    sys.stdout.close()


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
