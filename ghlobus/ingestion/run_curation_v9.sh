#! /bin/bash
# run_curation_v9.sh
#
# Processes exam and instance data and creates a merged instance table with metadata.
# for these datasets:
# FAMLI2_enrolled
# FAMLI2
# FAMLI3
#
# The following steps are involved in data curation (in order):
# 1. tablify_sr_v9.py: Creates tabular form the SR data; each SR file must be run separately
# 2. merge_sr_crf_v9.py: Merges SR with CRF data into a single consolidated exam table
# 3. merge_instance_exam_v9.py: Merges instance with exam data into instance table with metadata
# 4. preprocess_data_v9.py:
#   (a) Preprocesses the video data (DICOM and MP4) into PyTorch tensor files while
#       also checking to make sure each video exists
#   (b) Updates the instance table with the PyTorch tensor file paths and selected metadata
#   (c) Saves the 5th frame of each video as a PNG file
#   (d) Saves all the frames in a parallel directory structure (for SSL training)
#   (e) Filters out RGB Doppler videos
#   (f) Records the reason each video was rejected or unreadable
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

# Ensure the root of the repository is in the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/workspace/code

# FAMLI2 curation
python tablify_sr_v9.py --yaml configs/FAMLI2_sr_GE_v9.yaml
python merge_sr_crf_v9.py --yaml configs/FAMLI2_sr_crf_v9.yaml
python merge_instance_exam_v9.py --yaml configs/FAMLI2_IT_exam_v9.yaml
# FAMLI2 preprocessing (this step may take several hours)
python preprocess_data_v9.py --yaml configs/FAMLI2_preproc_v9.yaml
python merge_tag_v9.py --yaml configs/FAMLI2_tag_v9.yaml

# FAMLI2_enrolled curation
python tablify_sr_v9.py --yaml configs/FAMLI2_enrolled_sr_GE_Clarius_v9.yaml
python tablify_sr_v9.py --yaml configs/FAMLI2_enrolled_sr_Sonosite_v9.yaml
python merge_sr_crf_v9.py --yaml configs/FAMLI2_enrolled_sr_crf_v9.yaml
python merge_instance_exam_v9.py --yaml configs/FAMLI2_enrolled_IT_exam_v9.yaml
# FAMLI2_enrolled preprocessing (this step may take more than a day)
python preprocess_data_v9.py --yaml configs/FAMLI2_enrolled_preproc_v9.yaml
python merge_tag_v9.py --yaml configs/FAMLI2_enrolled_tag_v9.yaml

# FAMLI3 curation
python tablify_sr_v9.py --yaml configs/FAMLI3_sr_GE_v9.yaml
python merge_sr_crf_v9.py --yaml configs/FAMLI3_sr_crf_v9.yaml
python merge_instance_exam_v9.py --yaml configs/FAMLI3_IT_exam_v9.yaml
# FAMLI3 preprocessing (this step may take up to a day)
python preprocess_data_v9.py --yaml configs/FAMLI3_preproc_v9.yaml
python merge_tag_v9.py --yaml configs/FAMLI3_tag_v9.yaml

# EFW data curation
python efw_data_selection_v9.py --yaml configs/FAMLI2_enrolled_efw_data_v9.yaml
python efw_data_selection_v9.py --yaml configs/FAMLI2_efw_data_v9.yaml
python efw_data_selection_v9.py --yaml configs/FAMLI3_efw_data_v9.yaml
python efw_data_split_v9.py --yaml configs/efw_tvh_split_v9.yaml

# TWIN data curation
python twin_data_selection_v9.py --yaml configs/FAMLI2_enrolled_twin_data_v9.yaml
python twin_data_selection_v9.py --yaml configs/FAMLI2_twin_data_v9.yaml
python twin_data_selection_v9.py --yaml configs/FAMLI3_twin_data_v9.yaml
python twin_data_split_v9.py --yaml configs/twin_5fold_split_v9.yaml
