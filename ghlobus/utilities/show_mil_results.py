"""
show_mil_results.py

A function that plots the results of an exam-level MIL inference run.

Author: Courosh Mehanian

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import argparse

from ghlobus.utilities.plot_utils import show_mil_results
from ghlobus.utilities.data_utils import read_spreadsheet_columns
from ghlobus.utilities.constants import SOFTMAX_POS


# run this script if it is the main program, not if importing a method
if __name__ == "__main__":
    # Note that path must be specified; there is no default
    # Default file points to validation exam-level results
    DEFAULT_FILE = "val_exam_predictions.csv"
    LABEL_COL = "TWIN"

    # Configure the ArgumentParser
    cli_description = "Plot results from TWIN results exam-level spreadsheet.\n"
    cli_description += "Usage:\n"
    cli_description += "  python show_mil_results.py --path /path/to/your/result/folder --file val_exam_predictions.csv"

    # Add arguments to be read from the command line
    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--file', default=DEFAULT_FILE, type=str)

    # Extract command line arguments
    args = parser.parse_args()

    # get the experiment name
    experiment_name = os.path.basename(args.path)

    # read the exam-level results
    fpath = os.path.join(args.path, args.file)
    df = read_spreadsheet_columns(fpath,
                                  sheet=None,
                                  rows=None,
                                  columns=[LABEL_COL, SOFTMAX_POS])

    # get the labels and scores
    labels = df[LABEL_COL].values.astype(float).astype(int)
    scores = df[SOFTMAX_POS].values.astype(float)

    # Plot the results
    show_mil_results(experiment_name,
                     labels,
                     scores,
                     args.path)
