#! /bin/bash
# run_curation_v4.sh
#
# Processes exam and instance data and creates a merged instance table with metadata.
# Performs this for GA_NEJME data.
#
# The following steps are involved in data curation (in order):
# 1. merge_instance_exam_v4.py: Merges instance with exam data into instance table with metadata
# 2. preprocess_data_v4.py:
#   (a) Preprocesses the DICOM video data into PyTorch tensor files while
#       also checking to make sure each video exists
#   (b) Updates the instance table with the PyTorch tensor file paths and selected metadata
#
# Notes:
#   - this script assumes that it is being run in a Docker container
#   - the preprocessing steps may take several hours to complete
#   - other processing steps are considerably faster
#
# Author: Courosh Mehanian
#
# This software is licensed under the MIT license. See LICENSE.txt in the root of
# the repository for details.
#
# Ensure the root of the repository is in the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/workspace/code

# Create the necessary prototype files with ground truth labels
python merge_instance_exam_v4.py --yaml configs/GA_NEJME_IT_exam_v4.yaml

# Preprocess the data for training, validation, and testing (these steps may take several hours)
python preprocess_data_v4.py --yaml configs/GA_NEJME_train_preproc_v4.yaml
python preprocess_data_v4.py --yaml configs/GA_NEJME_val_preproc_v4.yaml
python preprocess_data_v4.py --yaml configs/GA_NEJME_test_preproc_v4.yaml

# Create a 100% split for GA training, validation, and testing
python distributions_v4.py --yaml configs/ga_100_distrib_v4.yaml

# Create a 100% split for FP training, validation, and testing
python distributions_v4.py --yaml configs/fp_100_distrib_v4.yaml
