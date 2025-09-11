"""
constants.py

Various constants used for data processing and such.

They can be used standalone if the parameters are specified.

Author: Daniel Shea
        Courosh Mehanian
        Sourabh Kulhare
        Olivia Zahn

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
BAD_SHAPE = (-99, -99, -99, -99)
BAD_BBOX = (-99, -99, -99, -99)
GOOD_VIDEO_MSG = "Good_video"
META_DIR = "sheets"
DIST_DIR = "distributions"
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"
RAW_DIR = "frame5"
RAW_FRAME_INDEX = 5
BATCH_SIZE = 1000
DICOM_FILE_TYPE = "dcm"
MP4_FILE_TYPE = "mp4"
PNG_FILE_TYPE = "png"
EXCLUDE_LABEL = -99
UNKNOWN_LABEL = -88
DEFAULT_VIDEO_PREDICTION_THRESHOLD = 0.45
VERY_LARGE_NUMBER = 1e8
GA_BINS = 4
GA_LOW = 40
GA_HIGH = 280
GA_WIDTH = (GA_HIGH - GA_LOW) // GA_BINS
SOFTMAX_NEG = "Softmax negative class"
SOFTMAX_POS = "Softmax positive class"
