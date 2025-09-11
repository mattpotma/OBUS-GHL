"""
distributions_v4.py

This module creates distributions containing stratified
 or balanced samples using a given fraction of the data,
 and a balance weight between 0 and 1.
 balance = 0 pure stratified, i.e., matching source distribution
 balance = 1 pure balance, i.e., all distribution bins are equal
 strategy = 'over' only use oversampling
 strategy = 'under' only use undersampling
 strategy = 'combo' use over- and under-sampling to achieve target count
 strategy = 'all' use all samples without balancing or stratification

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd

from ghlobus.utilities.constants import DIST_DIR
from ghlobus.utilities.constants import TRAIN_CSV
from ghlobus.utilities.constants import VAL_CSV
from ghlobus.utilities.constants import TEST_CSV
from ghlobus.utilities.sweep_utils import VERTICAL_SWEEP_TAGS
from ghlobus.utilities.sweep_utils import VERTICAL_PLUS_TRANSVERSE
from ghlobus.utilities.sample_utils import balanced_sample
from ghlobus.utilities.sample_utils import construct_bins

def main():
    """
    Main function to create distributions of training and validation data.

    Writes train.csv and val.csv files in the specified distribution directory.
    """
    # Default yaml file
    DEFAULT_YAML = "configs/ga_100_distrib_v4.yaml"

    # Configure the ArgumentParser
    cli_description = 'Develop distribution of train and val data at given fraction.\n'
    cli_description += 'Attempt stratified or balanced sampling of feature-of-interest.'

    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--yaml', default=DEFAULT_YAML, type=str)

    # Extract command line arguments
    args = parser.parse_args()

    # read parameter yaml file
    with open(args.yaml, 'r') as f:
        info = yaml.safe_load(f)

    # Get parameters to use for distribution
    src_dir = info['input']['src_dir']
    dest_dir = info['output']['dest_dir']
    dist_name = info['output']['dist_name']
    # Fraction of data to use for distribution
    fraction = info['output']['fraction']
    # Filtering on a label (reject negative values)
    label = info['filtering']['label']
    # Filtering on vertical sweep tags
    if 'sweeps' in info['filtering']:
        if info['filtering']['sweeps'] is not None:
            vertical_filtering = True
            vertical_dev = info['filtering']['sweeps']['dev']
            vertical_test = info['filtering']['sweeps']['test']
        else:
            vertical_filtering = False
            vertical_dev = None
            vertical_test = None
    else:
        vertical_filtering = False
        vertical_dev = None
        vertical_test = None
    # Filtering on a continuous feature
    if 'feature' in info['filtering']:
        # See if it is specified
        if info['filtering']['feature'] is not None:
            # Get feature name and limits
            filter_feature = info['filtering']['feature']['column']
            low = info['filtering']['feature']['low']
            high = info['filtering']['feature']['high']
        else:
            # Not specified, set to null
            filter_feature = None
            low = None
            high = None
    else:
        # Not specified, set to null
        filter_feature = None
        low = None
        high = None
    # Balancing parameters
    balance_feature = info['balancing']['feature']
    feature_type = info['balancing']['type']
    nbins = info['balancing']['nbins']
    balance = info['balancing']['balance']
    strategy = info['balancing']['strategy']

    # Generate name of distribution folder
    dist_dir = os.path.join(dest_dir, DIST_DIR, str(dist_name))

    # print information about preprocessing run
    print_string = f"Distributions will be written here {dist_dir},\n"
    print_string += f"Using {fraction*100}% of the data.\n"
    if label:
        print_string += f"Feature {label} will be filtered for positive values.\n"
    if filter_feature:
        print_string += f"Feature {filter_feature} will be filtered between {low} and {high}.\n"
    print_string += f"Feature {balance_feature} will be balanced on.\n"
    print(print_string)

    # create distribution directory if not there
    if not os.path.exists(dist_dir):
        # directory not there, create it
        print(f"Creating folder {dist_dir}.")
        os.makedirs(dist_dir, exist_ok=True)
    else:
        # directory is there, see if TRAIN_CSV or VAL_CSV are there
        if (os.path.exists(os.path.join(dist_dir, TRAIN_CSV)) or
                os.path.exists(os.path.join(dist_dir, VAL_CSV))):
            raise ValueError(f"Folder {dist_dir} exists and distributions exist."
                             + f" Choose new index value. Aborting.")
        else:
            print(f"Distribution folder {dist_dir} exists.")

    # Make dictionary of sets to process
    sets = dict()
    # Create 2-member tuples of file paths where the
    # elements are (input_file, distribution_file)
    sets['Train'] = (
        os.path.join(src_dir, TRAIN_CSV),
        os.path.join(dist_dir, TRAIN_CSV),
    )
    sets['Val'] = (
        os.path.join(src_dir, VAL_CSV),
        os.path.join(dist_dir, VAL_CSV),
    )
    sets['Test'] = (
        os.path.join(src_dir, TEST_CSV),
        os.path.join(dist_dir, TEST_CSV),
    )

    # Iterate through sets to create fractional distribution
    for subset, file in sets.items():
        # read data summary
        set_df = pd.read_csv(file[0])
        # Check if we are to filter by label
        if label is not None:
            set_df = set_df[set_df[label] >= 0]
        # Check if we are to filter by vertical sweep tags
        if vertical_filtering:
            # Set the sweep filtering directive based on subset
            if subset == 'Train' or subset == 'Val':
                # For the development set, use sweep types indicated by vertical_dev
                filtering_directive = vertical_dev
            else:
                # For the test set, use sweep types indicated by vertical_test
                filtering_directive = vertical_test
            # Apply the filtering based on the specified types
            if filtering_directive == 'vertical':
                # Keep only vertical sweep tags
                set_df = set_df[set_df['tag'].isin(VERTICAL_SWEEP_TAGS)]
            elif filtering_directive == 'transverse':
                # Keep vertical and transverse sweep tags
                set_df = set_df[set_df['tag'].isin(VERTICAL_PLUS_TRANSVERSE)]
            else:
                # no sweep filtering
                pass
        # Check if we are to filter by feature
        if filter_feature:
            set_df = set_df[(set_df[filter_feature] >= low) &
                            (set_df[filter_feature] <= high)]
        # check the balance strategy
        if subset == 'Test' or strategy == 'all':
            # No balancing, use all samples
            distrib_inds = np.arange(set_df.shape[0])
        else:
            # We are balancing, get the feature of interest
            bal_ftr = set_df[balance_feature].values
            # Check if feature is continuous or categorical
            if feature_type == 'continuous':
                # continuous feature, we need to create bins
                if nbins is None:
                    raise ValueError("Number of bins must be specified for continuous features.")
                # only calculate ftr_bins for Train set
                elif subset == 'Train':
                    # create quantization bins based on quantiles
                    ftr_bins = construct_bins(bal_ftr, nbins, balance)
                # digitize the feature
                # noinspection PyUnboundLocalVariable
                y = np.digitize(bal_ftr, ftr_bins) - 1
            else:
                # categorical feature, we can use it directly
                y = bal_ftr
            # generate random split, balancing by y
            distrib_inds = balanced_sample(y,
                                           ftr=None,
                                           fraction=fraction,
                                           balance=balance,
                                           strategy=strategy
                                           )
        # grab the sampled rows of the dataframe
        dist_df = set_df.iloc[distrib_inds]
        # write distribution csv
        dist_df.to_csv(file[1], index=False)
        print(f"Wrote {file[1]}.")


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    main()
