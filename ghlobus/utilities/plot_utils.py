"""
plot_utils.py

Utilities related to plotting of inference results.

They can be used standalone if the parameters are specified.

Author: Daniel Shea
		Courosh Mehanian
        Wenlong Shi

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from typing import Literal, List, Tuple, Union

from ghlobus.utilities.sweep_utils import KNOWN_COMBO_TAGS
from ghlobus.utilities.sweep_utils import BLIND_SWEEP_TAGS
from ghlobus.utilities.constants import SOFTMAX_POS

# Gestational age limits
ga_limits = (60, 280)
ga_ticks = np.arange(63, 274, 21)

# Some style settings
err_ticks = np.arange(-21, 22, 7)

# definition of trimesters
trimesters = [(40, 13*7-1), (13*7, 28*7-1), (28*7, 300)]

# Define scatter plot options for each of the dataset splits
scatter_plot_args = {
    'train': {
        'alpha': 0.5,
        'color': 'blue',
        's': 5,
    },
    'val': {
        'alpha': 0.75,
        'color': 'orange',
        's': 5,
    },
    'test': {
        'alpha': 0.75,
        'color': 'green',
        's': 5,
    },
}


def plot_dataset_truth_vs_predictions(df: pd.DataFrame,
                                      set_name: Literal["train",
                                                        "val", "test"] = 'train',
                                      label_name: str = 'ga_boe',
                                      legend: bool = True,
                                      title: Union[str, None] = None,
                                      label_plot_name: str = "GA",
                                      unit_plot_name: str = "Days",
                                      ) -> Figure:
    """
    Plots results data from a Pandas DataFrame for a data subset ('train', 'val', 'test').
    Plots the true value vs the predicted value as a scatter plot.

    Parameters:
        df: pd.DataFrame      DF containing the dataset to plot
        set_name: str         Name of the dataset subset ('train', 'val', 'test')
        label_name: str       Name of the label column in the DataFrame
        legend: bool          Whether to show the legend in the plot
        title: str            Title of the plot (optional)
        label_plot_name: str  Name of the label visualized in the plots
        unit_plot_name: str   Unit of the label visualized in the plots

    Returns:
    - fig: Matplotlib Figure  Object representing the plot
    """
    true = df[label_name].tolist()
    pred = df[f'Predicted {label_plot_name} ({unit_plot_name})'].tolist()

    # Plot the Prediction vs Truth line, and give it a label
    fig = plt.figure()
    if title:
        plt.title(label=title, fontsize=14)
    plt.scatter(true, pred, label=set_name, **scatter_plot_args[set_name])
    # Plot a truth line
    min_val = min(min(true), min(pred))
    max_val = max(max(true), max(pred))
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle='--', color='k', label='Truth')
    plt.xlabel(f'True {label_plot_name} ({unit_plot_name})', fontsize=14)
    plt.ylabel(f'Predicted {label_plot_name}', fontsize=14)
    if legend:
        plt.legend()
    return fig


def plot_dataset_fractional_bland_altman(df: pd.DataFrame,
                                         set_name: Literal["train",
                                                           "val", "test"] = 'train',
                                         label_name: str = 'efw_hadlock',
                                         err_col: str = 'Fractional Error',
                                         plot_ranges: tuple = (),
                                         title: str = None,
                                         label_plot_name: str = "EFW",
                                         unit_plot_name: str = "Grams",
                                         ) -> Figure:
    """
    Plots a fractional Bland-Altman plot for a given dataset, including the 5th and 95th
    percentiles or efw <= 3000 or > 3000.

    Parameters:
    - df:               Pandas DataFrame containing the dataset to plot
    - set_name:         Name of the dataset subset ('train', 'val', 'test')
    - label_name:       Name of the label column in the DataFrame
    - err_col:          Column containing prediction error
    - title:            Title of the plot (optional)
    - label_plot_name:  Name of the label to plot (optional)
    - unit_plot_name:   Unit of the label to plot (optional)
    - plot_ranges:      List of tuples of ranges to plot the 5th and 95th percentiles (optional)

    Returns:
    - fig: Matplotlib Figure object representing the plot
    """
    # Extract the true and error values
    true = df[label_name].tolist()
    err = df[err_col].tolist()

    # Calculate MRAE
    mrae = np.mean(np.abs(err))
    # Calculate 5th and 95th percentiles
    y05, y95 = np.percentile(err, [5, 95])

    # Plot the Prediction vs Truth line, and give it a label
    fig = plt.figure()
    if title:
        plt.title(label=title)
    # Print MRAE statistic in the legend
    plt.scatter(true, err,
                label=f"{set_name}: MRAE: {mrae:0.3f}",
                **scatter_plot_args[set_name])

    # Plot the 5th and 95th percentile lines
    # Only show it if not plotting by range
    if not plot_ranges:
        xlim = plt.xlim()
        plt.plot(xlim, [y05, y05], 'r--', lw=1, label=f"5th prctl: {y05:0.3f}")
        plt.plot(xlim, [y95, y95], 'm--', lw=1, label=f"95th prctl: {y95:0.3f}")

    # plot 5 percentile and 95 percentile of err for data in different ranges, write the value on the plot
    for range_limits in plot_ranges:
        errors_in_range = [err[i] for i in range(
            len(true)) if range_limits[0] < true[i] <= range_limits[1]]
        if errors_in_range:
            error_percentile_5 = np.percentile(errors_in_range, 5)
            error_percentile_95 = np.percentile(errors_in_range, 95)
            plt.plot([range_limits[0], range_limits[1]], [
                     error_percentile_5, error_percentile_5], 'k--', lw=1, c='r')
            plt.plot([range_limits[0], range_limits[1]], [
                     error_percentile_95, error_percentile_95], 'k--', lw=1, c='r')
            plt.text(range_limits[0], error_percentile_5,
                     f'{error_percentile_5*100:.2f}%', fontsize=16, c='r')
            plt.text(range_limits[0], error_percentile_95,
                     f'{error_percentile_95*100:.2f}%', fontsize=16, c='r')

    plt.xlabel(f'True {label_plot_name} ({unit_plot_name})')
    plt.ylabel(f'Fractional Error')
    plt.legend()
    return fig


def plot_dataset_bland_altman(df: pd.DataFrame,
                              set_name: Literal["train",
                                                "val", "test"] = 'train',
                              label_name: str = 'ga_boe',
                              err_col: str = 'Prediction Error (days)',
                              limits: Union[Tuple[float, float], None] = None,
                              by_trimester: bool = False,
                              legend: bool = True,
                              title: Union[str, None] = None,
                              label_plot_name: str = "GA",
                              unit_plot_name: str = "Days",
                              ) -> Figure:
    """
    Plots a Bland-Altman plot for a given dataset, including the 5th and 95th
    percentiles.

    Parameters:
        df: pd.DataFrame              Dataset containing the data to plot and prediction error.
        set_name: str                 Name of the dataset. Defaults to 'train'.
        label_name: str               Column name for the true gestational age.
        err_col: str                  Column containing prediction error.
                                      Defaults to 'Prediction Error, days'.
        limits: Tuple[float, float]   Limits for the plot. Defaults to None.
        by_trimester: bool            Whether to plot by trimester. Defaults to False.
        legend: bool                  Whether to show the legend. Defaults to True.
        title: Union[str, None]       Title of the plot. Defaults to None.
        label_plot_name: str          Name of the label visualized in the plots
        unit_plot_name: str           Unit of the label visualized in the plots

    Returns:
        Figure: The matplotlib Figure object representing the plot.
    """
    # Extract the true and error values
    true = df[label_name].values
    err = df[err_col].values

    # Calculate MAE
    mae = np.mean(np.abs(err))
    # Calculate 5th and 95th percentiles
    y05, y95 = np.percentile(err, [5, 95])

    # Plot the Prediction vs Truth line, and give it a label
    fig = plt.figure()
    if title:
        plt.title(label=title, fontsize=16)
    # Print MAE statistic in the legend
    plt.scatter(true, err,
                label=f"{set_name}: MAE: {mae:0.3f} {unit_plot_name}",
                **scatter_plot_args[set_name])
    plt.xlabel(f'True {label_plot_name} ({unit_plot_name})', fontsize=14)
    plt.ylabel(f'Prediction Error ({unit_plot_name})', fontsize=14)
    if limits:
        plt.ylim(limits)
        plt.xlim(ga_limits)
        plt.xticks(ga_ticks, ga_ticks, fontsize=14)
        plt.yticks(err_ticks, err_ticks, fontsize=14)
    # Plot the 5th and 95th percentile lines
    # Only show it if not plotting by_trimester
    if not by_trimester:
        xlim = plt.xlim()
        plt.plot(xlim, [y05, y05], 'r--', lw=1,
                 label=f"5th prctl: {y05:0.3f} {unit_plot_name}")
        plt.plot(xlim, [y95, y95], 'm--', lw=1,
                 label=f"95th prctl: {y95:0.3f} {unit_plot_name}")

    if by_trimester:
        xlimits = plt.xlim()
        for tri in trimesters:
            tri_err = err[(df['ga_boe'] > tri[0]) & (df['ga_boe'] <= tri[1])]
            y05, y95 = np.percentile(tri_err, [5, 95])
            xmin = max(tri[0], xlimits[0])
            xmax = min(xlimits[1], tri[1])
            xmid = (xmin + xmax) / 2
            plt.plot([tri[0], tri[1]], [y05, y05],
                     color='firebrick', linestyle='--', linewidth=1)
            plt.text(x=xmid, y=y05 - 3, s=f"{y05:0.2f}",
                     fontsize=12, fontweight='bold', color='firebrick')
            plt.plot([tri[0], tri[1]], [y95, y95],
                     color='firebrick', linestyle='--', linewidth=1)
            plt.text(x=xmid, y=y95 + 2, s=f"{y95:0.2f}",
                     fontsize=12, fontweight='bold', color='firebrick')
    if legend:
        plt.legend()
    return fig


def compute_prediction_loss(df: pd.DataFrame,
                            loss: torch.nn.Module = torch.nn.L1Loss()) \
        -> torch.Tensor:
    """
    Compute the prediction loss using the specified loss function.

    Parameters:
        df (pandas.DataFrame):             Input DataFrame containing the predicted and true values.
        loss (torch.nn.Module, optional):  Loss function to use. Defaults to torch.nn.L1Loss().

    Returns:
        torch.Tensor: The computed loss value.
    """
    # Compute the loss
    pred = df['Predicted GA (Days)'].tolist()
    true = df['ga_boe'].tolist()
    loss_result = loss(torch.Tensor(pred), torch.Tensor(true))
    return loss_result


def plot_by_trimester(df,
                      set_name='train',
                      title: str = None,
                      loss=torch.nn.L1Loss(),
                      ) -> List[Figure]:
    """
    Plots the predicted gestational age (GA) against the true GA for each trimester.

    Parameters:
        df (DataFrame):                     Input DataFrame containing the data.
        set_name (str, optional):           Name of the data set. Defaults to 'train'.
        title (str, optional):              Title for the plot. Defaults to None.
        loss (torch.nn.Module, optional):   Loss function to calculate the trimester loss.
                                            Defaults to torch.nn.L1Loss().

    Returns:
        List[Figure]:                       List of matplotlib Figure objects,
                                            one for each trimester.

    """
    # Trimester analysis
    # NOTE -- this does NOT separate novice out
    first_trimester = df['ga_boe'] <= 90
    second_trimester = (df['ga_boe'] > 90) & (df['ga_boe'] <= 196)
    third_trimester = df['ga_boe'] > 196
    trimesters = [first_trimester, second_trimester, third_trimester]
    titles = ['First Trimester', 'Second Trimester', 'Third Trimester']

    # Add the supplied title to the titles in the list, if provided
    if title:
        titles = [f"{title}: {x}" for x in titles]

    # An empty list to catch the matplotlib plots.
    figs = []
    for trimester, title in zip(trimesters, titles):
        # Get the rows for this specific trimester
        trimester_rows = df.loc[trimester]
        # Extract truth and prediction lists
        pred = trimester_rows['Predicted GA (Days)'].tolist()
        true = trimester_rows['ga_boe'].tolist()

        # Check if the trimester is empty
        if len(trimester_rows.index) == 0:
            # append a None for this trimester figure
            figs.append(None)
            continue

        # Recalculate the loss for this specific trimester/data set
        trimester_loss = loss(torch.Tensor(pred), torch.Tensor(true))

        # Determine which set of args to use:
        scatter_kwargs = scatter_plot_args[set_name]

        # Plot the Prediction vs Truth line, and give it a label
        fig = plt.figure()
        plt.title(title)
        plt.scatter(true, pred, label=set_name, **scatter_kwargs)
        # Plot a truth line
        plt.plot([min(true)-3, max(true)+3], [min(true)-5, max(true)+5],
                 linestyle='--', color='k', label='Truth')
        # Place the loss in MAE (days) on the plot
        ax = plt.gca()
        plt.text(x=0.25, y=0.75, s='MAE (days): {:.4f}'.format(
            trimester_loss), ha='center', transform=ax.transAxes)
        # Label axes
        plt.ylabel('Predicted Gestational Age (days)')
        plt.xlabel('True Gestational Age (days)')
        plt.legend()
        # Print an output, then show the plot
        print(set_name, title, trimester_loss, len(trimester_rows.index))
        figs.append(fig)
    # Return the 3 figures
    return figs


def plot_attention_scores(attention_scores: np.ndarray,
                          video_name: str,
                          ) -> Figure:
    """
    Plot attention score vs (concatenated) frame number.

    Parameters:
        attention_scores (np.ndarray):   Attention scores to plot.
        video_name (str):                Name of the video.

    Returns:
        Figure: The matplotlib Figure object with the plotted data.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.squeeze(attention_scores))
    ax.set_title(video_name)
    ax.set_ylabel('Attention Score')
    ax.set_xlabel('Frame Number')
    return fig


def plot_exam_attention_scores(attention_scores: np.ndarray,
                               video_lengths: List[int],
                               video_names: List[str],
                               ) -> Figure:
    """
    Plot attention score vs (concatenated) frame number.

    Parameters:
        attention_scores (np.ndarray):   Attention scores to plot.
        video_lengths (List[int]):       Lengths of the videos.
        video_names (List[str]):         Names of the videos.

    Returns:
        Figure: The matplotlib Figure object with the plotted data.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.squeeze(attention_scores))

    start_index = 0
    for (i, width), video_name in zip(enumerate(video_lengths), video_names):
        end_index = start_index + width
        mid_index = (start_index + end_index)/2
        if i % 2 == 0:
            ax.axvspan(start_index, end_index, alpha=0.5, lw=0)
        ax.text(mid_index, 0.0028, video_name, rotation=45)
        start_index = end_index

    ax.set_ylabel('Attention Score')
    ax.set_xlabel('Frame Number')
    return fig


def plot_auroc_curve(df: pd.DataFrame,
                     title: str = None,
                     label_col: str = 'lie',
                     ) -> Figure:
    """
    Plot the ROC curve for a binary classifier.

    Parameters:
        df (pd.DataFrame):          Input DataFrame containing the data.
        title (str, optional):      Title of the plot. Defaults to None.
        label_col (str, optional):  Column name of the label column. Defaults to 'lie'.

    Shows:
        fpr (np.ndarray): The false positive rate.
            vs.
        tpr (np.ndarray): The true positive rate.
            and the AUC value.
        roc_auc (float): The area under the ROC curve.

    Returns:
        Figure: The matplotlib Figure object with the plotted data.
    """
    labels = df[label_col]
    pos_probs = df[SOFTMAX_POS]

    fpr, tpr, threshold = roc_curve(labels, pos_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return fig


def distribution_stats(distributions, save_path, feature_col_name):
    """
    Generate and save distribution statistics for binary features
    across train/val/test datasets.

    This function analyzes binary feature distributions across
    multiple datasets (typically train, validation, and test sets),
    prints statistics to console, saves them to a file, and creates
    histogram plots for positive and negative cases separately.

    Parameters:
        distributions:     List of DataFrames for train, val, and test sets
        save_path:         Directory path where output files will be saved
        feature_col_name:  Name of the binary feature column to analyze

    Returns:
        None

    Note:
        Function assumes datasets contain 'PID' and 'GA' columns and only supports
        binary features (values 0 and 1).
    """
    dist_order = ['train', 'val', 'test']
    dist_colors = ['blue', 'green', 'red']
    for dist_idx, dist_df in enumerate(distributions):
        # print the statistics of the train, val, and test dataframes
        print(
            f"{dist_order[dist_idx]} TWIN PID counts: {dist_df.groupby('PID')[feature_col_name].first().value_counts().values}")
        print(
            f"{dist_order[dist_idx]} Video counts : {dist_df[feature_col_name].value_counts().values} ")

    # save the statistics to a file
    with open(os.path.join(save_path, 'counts.txt'), 'w') as f:
        for dist_idx, dist_df in enumerate(distributions):
            f.write(
                f"{dist_order[dist_idx]} TWIN PID counts: {dist_df.groupby('PID')[feature_col_name].first().value_counts().values}\n")
            f.write(
                f"{dist_order[dist_idx]} Video counts : {dist_df[feature_col_name].value_counts().values}\n")
            f.write("\n")

        f.close()

    # plot the GA values for train, val, and test for TWIN = 1
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for dist_idx, dist_df in enumerate(distributions):

        dist_pos_df = dist_df[dist_df[feature_col_name] == 1]

        sns.histplot(dist_pos_df['GA'], ax=ax[dist_idx],
                     color=dist_colors[dist_idx], label=dist_order[dist_idx] + ' positive')
        ax[dist_idx].set_title(
            f"{dist_order[dist_idx]} {feature_col_name} distribution")
        ax[dist_idx].set_xlabel('GA')
        ax[dist_idx].set_ylabel('Video Counts')

    plt.savefig(os.path.join(save_path, 'positive_distribution.png'))

    # plot the GA values for train, val, and test for TWIN = 0
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for dist_idx, dist_df in enumerate(distributions):

        dist_neg_df = dist_df[dist_df[feature_col_name] == 0]

        sns.histplot(dist_neg_df['GA'], ax=ax[dist_idx],
                     color=dist_colors[dist_idx], label=dist_order[dist_idx] + ' negative')
        ax[dist_idx].set_title(
            f"{dist_order[dist_idx]} {feature_col_name} distribution")
        ax[dist_idx].set_xlabel('GA')
        ax[dist_idx].set_ylabel('Video Counts')

    plt.savefig(os.path.join(save_path, 'negative_distribution.png'))


def accuracy_heatmaps_by_count_and_score(filepath: str) -> List[Figure]:
    """
    Plot heatmaps of sensitivity and specificity vs score and video thresholds.

    Args:
        filepath (str):  Path to the file containing the video output scores.

    Returns:
        List[Figure]:    List of matplotlib Figure objects with heatmaps.
    """
    # read the output results
    df = pd.read_csv(filepath, index_col=0)

    # rename score columns for clarity
    df = df.rename(columns={SOFTMAX_POS: 'twin_score'})

    # organize exams by study ID
    exams = df['StudyID'].values
    uniq_exams = df['StudyID'].unique()

    # process one exam at a time
    selected_exams = list()
    selected_tags = list()
    selected_scores = list()
    twins = list()
    blind_counts = list()
    total_counts = list()
    for exam in uniq_exams:
        # get exam mask
        exam_mask = exams == exam
        # ground truth for exam
        TWINs = df.loc[exam_mask, 'TWIN'].values
        # check that all videos in exam have same TWIN value
        if len(np.unique(TWINs)) != 1:
            # inconsistent TWIN values
            print(f'Exam {exam} has inconsistent TWIN values')
            continue
        # get scores for exam
        scores = df.loc[exam_mask, 'twin_score'].values
        # get tags for exam
        tags = df.loc[exam_mask, 'tag'].values
        # check for blind sweeps
        bad = np.isin(tags, BLIND_SWEEP_TAGS, invert=True)
        if bad.all():
            # no blind sweeps in exam
            print(f'Exam {exam} has no blind sweeps')
            continue
        # get count of video statistics
        total_counts.append(len(tags))
        blind_counts.append(len(tags[~bad]))
        # append TWIN label
        twins.append(TWINs[0])
        selected_exams.append(exam)
        # see if there is a tag combination that matches one of the known combos
        for kc in KNOWN_COMBO_TAGS:
            if np.isin(list(kc), tags).all():
                # this known combo is present in tags
                # find indices in tags that match
                indices = [np.where(tags == tag)[0][0] for tag in kc]
                break
        else:
            # known combo not found, limit to blind sweeps
            print(f'Exam {exam} has no known tag combinations')
            tags = tags[~bad]
            scores = scores[~bad]
            if len(tags) < 6:
                # less than 6; must duplicate some
                indices = list(range(len(tags))) + \
                    list(np.random.choice(len(tags), 6 - len(tags), replace=True))
            elif len(tags) > 6:
                # more than 6; sample 6 random videos
                indices = np.random.choice(len(tags), 6, replace=False)
            else:
                # there are exactly 6, take all
                indices = range(6)
        # append selected count, tags, scores
        selected_tags.append(tags[indices])
        selected_scores.append(scores[indices])

    # convert to dataframe and save
    data_dict = {
        'StudyID': selected_exams,
        'TWIN': twins,
        'total_count': total_counts,
        'blind_count': blind_counts,
        'tag': selected_tags,
    }
    # add score columns to data_dict
    twins = np.array(twins)
    selected_scores = np.array(selected_scores)
    for i in range(6):
        data_dict[f"score{i}"] = selected_scores[:, i]
    exam_df = pd.DataFrame(data_dict)
    # save to csv, get root directory
    root = os.path.dirname(filepath)
    exam_df.to_csv(os.path.join(
        root, 'exam_known_combo_scores.csv'), index=False)

    # generate heatmaps
    score_thresholds = np.round(np.linspace(0.1, 0.9, 9), decimals=1)
    video_thresholds = range(1, 7)
    sensitivity = np.zeros((len(video_thresholds), len(score_thresholds)))
    specificity = np.zeros((len(video_thresholds), len(score_thresholds)))
    for iy, vt in enumerate(video_thresholds):
        for ix, st in enumerate(score_thresholds):
            # get mask for scores above threshold
            mask = selected_scores >= st
            # get mask for videos with at least vt videos above threshold
            mask = mask.sum(axis=1) >= vt
            # get mask for TWINs
            twin_mask = twins.astype(bool)
            # calculate sensitivity
            sensitivity[iy, ix] = mask[twin_mask].mean()
            # calculate specificity
            specificity[iy, ix] = (~mask[~twin_mask]).mean()

    # plot heatmaps
    figs = list()
    fig = plt.figure()
    plt.imshow(sensitivity, origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel('Score Threshold')
    plt.ylabel('Video Threshold')
    plt.title('Sensitivity')
    plt.xticks(range(len(score_thresholds)), score_thresholds, fontsize=10)
    plt.yticks(range(len(video_thresholds)), video_thresholds, fontsize=10)
    # superimpose pixel values on image
    for iy in range(len(video_thresholds)):
        for ix in range(len(score_thresholds)):
            if sensitivity[iy, ix] < 0.75:
                color = 'w'
            else:
                color = 'k'
            plt.text(ix, iy, f"{sensitivity[iy, ix]:.2f}",
                     ha='center', va='center', color=color, fontsize=6)
    plt.show()
    figs.append(fig)
    fig = plt.figure()
    plt.imshow(specificity, origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel('Score Threshold')
    plt.ylabel('Video Threshold')
    plt.title('Specificity')
    plt.xticks(range(len(score_thresholds)), score_thresholds, fontsize=10)
    plt.yticks(range(len(video_thresholds)), video_thresholds, fontsize=10)
    # superimpose pixel values on image
    for iy in range(len(video_thresholds)):
        for ix in range(len(score_thresholds)):
            if specificity[iy, ix] < 0.75:
                color = 'w'
            else:
                color = 'k'
            plt.text(ix, iy, f"{specificity[iy, ix]:.2f}",
                     ha='center', va='center', color=color, fontsize=6)
    plt.show()
    figs.append(fig)
    return figs


def plot_histogram_by_group(df: pd.DataFrame,
                            column: str,
                            feature: str,
                            bins_range: tuple = (40, 260, 20)) -> None:
    """
    Plot histograms of a specified feature in the DataFrame, grouped by a column,
    with each group's histogram displayed in a subplot.

    Args:
        df (pd.DataFrame):   Input DataFrame.
        column (str):        Column to group by.
        feature (str):       Feature to plot histograms for.
        bins_range (tuple):  Start, stop, and step limits for histogram bins.

    Returns:
        None
    """
    # Group the DataFrame by the specified column
    grouped = df.groupby(column)
    group_names = list(grouped.groups.keys())

    # Compute subplot layout
    n_groups = len(group_names)
    n_rows, n_cols = tile_figure(n_groups)

    # Bins for the histograms
    bins = np.arange(*bins_range)

    # Reshape the groups for tiling
    group_names = np.array(group_names).reshape(n_rows, n_cols)
    group_data = np.empty((n_rows, n_cols), dtype=object)
    max_count = 0
    for g in grouped:
        loc = np.where(group_names == g[0])
        group_data[loc[0][0], loc[1][0]] = g[1]
        histo, _ = np.histogram(g[1][feature].astype(float), bins=bins)
        max_count = max(max_count, np.max(histo))
    max_count = nicelimit(max_count)

    # Create the sub-plot figure
    fig, axes = plt.subplots(nrows=n_rows,
                             ncols=n_cols,
                             figsize=(8 * n_cols, 6 * n_rows),
                             constrained_layout=True)

    # Plot each group's histogram in a subplot
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            name = group_names[row][col]
            df = group_data[row][col]
            ax.hist(df[feature].astype(float), bins=bins,
                    alpha=0.9, color='royalblue', edgecolor='none')
            ax.set_title(f"{feature} histogram for {name}", fontsize=20)
            ax.set_xlabel(feature, fontsize=16)
            ax.set_xlim(40, 260)
            ax.set_xticks(bins)
            ax.set_ylim(0, max_count)
            ax.tick_params(labelsize=14)
            ax.grid(True)

    # Show the plot
    plt.show()


def show_mil_results(experiment_name: str,
                     labels: np.ndarray,
                     scores: np.ndarray,
                     path: os.PathLike,
                     attention_scores: Union[None, np.ndarray] = None,
                     ) -> Figure:
    """
    Parameters
    ----------
        experiment_name: str          - Name of the experiment
        labels: np.ndarray            - True labels of the data
        scores: np.ndarray            - Predicted scores
        path: os.PathLike             - Path to save the results
        attention_scores: np.ndarray  - Attention scores for the data (optional)

    Returns
    -------
        fig: matplotlib Figure object with the plotted results.
    """
    # Create path for console output
    logger_path = os.path.join(path,
                               f"{experiment_name}_result.txt")
    # Set up the logger
    sys.stdout = Logger(logger_path)

    # calculate the roc auc score
    roc_auc = roc_auc_score(labels, scores)
    print(f"ROC AUC score: {roc_auc:0.4f}")

    # plot the ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)

    # set up the output figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
    axes[0, 0].plot(fpr, tpr)
    axes[0, 0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0, 0].set_title(f"AUC-ROC = {roc_auc:0.3f}")
    axes[0, 0].grid(True)

    # get the sensitivity and specificity at different thresholds
    thresholds = np.linspace(0, 1, 101)
    sensitivities = []
    specificities = []
    for threshold in tqdm(thresholds):
        predictions = scores > threshold
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # get the best threshold
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    dist2_to_UR_corner = np.power(
        1-sensitivities, 2) + np.power(1-specificities, 2)
    best_threshold = thresholds[np.argmin(dist2_to_UR_corner)]
    print(f"Best threshold: {best_threshold:0.2f}")

    # plot the sensitivity and specificity vs threshold
    axes[1, 0].plot(thresholds, sensitivities, label='Sensitivity')
    axes[1, 0].plot(thresholds, specificities, label='Specificity')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title(f"Best threshold: {best_threshold:0.2f}")
    axes[1, 0].legend()
    ylims = plt.ylim()
    axes[1, 0].plot([best_threshold]*2, ylims, '--', color='gray')
    axes[1, 0].grid(True)

    # get the sensitivity and specificity at the best threshold
    predictions = scores > best_threshold
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"Sensitivity at best threshold: {sensitivity:0.4f}")
    print(f"Specificity at best threshold: {specificity:0.4f}")

    # Create a custom colormap
    cmap = ListedColormap(['#FF9999', '#99CCFF'])

    # plot the confusion matrix
    sns.heatmap(ax=axes[0, 2],
                data=cm,
                annot=True,
                annot_kws={"size": 24},
                fmt='d',
                cmap=cmap,
                cbar=False)
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('Truth')
    axes[0, 2].set_title(f"Confusion matrix @ {best_threshold:0.2f}")
    axes[0, 2].text(0.1, 0.9, f"Specificity: {specificity:0.3f}", fontsize=16)
    axes[0, 2].text(1.1, 1.9, f"Sensitivity: {sensitivity:0.3f}", fontsize=16)

    # calculate the accuracy
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    print(f"Accuracy: {accuracy:0.4f}")

    # Assuming scores and labels are defined
    label0 = labels == 0
    label1 = labels == 1
    # get positive and negative exam counts
    n_positive = np.sum(labels)
    n_negative = len(labels) - n_positive
    bins = np.linspace(0, 1, 21)
    sns.histplot(ax=axes[0, 1],
                 data={'scores': scores[label0]},
                 x='scores',
                 color='red',
                 bins=bins,
                 kde=True)
    axes[0, 1].set_title(f"Singleton score histograms")
    axes[0, 1].set_xlim(-0.05, 1.05)
    ylims = axes[0, 1].get_ylim()
    axes[0, 1].text(0.4, np.mean(ylims)+0.3*np.diff(ylims)[0],
                    f"Negative exams: {n_negative}", fontsize=16)
    sns.histplot(ax=axes[1, 1],
                 data={'scores': scores[label1]},
                 x='scores',
                 color='blue',
                 bins=bins,
                 kde=True)
    axes[1, 1].set_title(f"Twin score histograms")
    axes[1, 1].set_xlim(-0.05, 1.05)
    ylims = axes[1, 1].get_ylim()
    axes[1, 1].text(0.1, np.mean(ylims)+0.2*np.diff(ylims)[0],
                    f"Positive exams: {n_positive}", fontsize=16)

    # Get a positive and a negative exam attention score profile
    if attention_scores is None:
        axes[1, 2].set_axis_off()
    else:
        neg_idx = np.where(label0)[0][0]
        pos_idx = np.where(label1)[0][0]
        neg_attention_scores = attention_scores[neg_idx]
        pos_attention_scores = attention_scores[pos_idx]

        # Plot the attention scores for both samples
        axes[1, 2].plot(neg_attention_scores, color='red',
                        label='Negative Sample')
        axes[1, 2].plot(pos_attention_scores, color='blue',
                        label='Positive Sample')
        axes[1, 2].set_title("Pos & Neg attention scores")
        axes[1, 2].set_xlabel('Frame')
        axes[1, 2].set_ylabel('Attention score')
        axes[1, 2].legend()

    # save the overall figure as a file
    plt.suptitle(f"Results for {experiment_name}")
    fig = plt.gcf()
    plt.savefig(os.path.join(path, f"{experiment_name}_results.png"))
    plt.close()

    # close stdout
    sys.stdout.close()

    return fig


def tile_figure(n_plots, plot_size=(1, 1)):
    """
    Calculate the number of rows and columns to tile a figure.

    Args:
        n_plots (int):      Number of plots.
        plot_size (tuple):  Size of each plot as (height, width).

    Returns:
        tuple:              Number of rows and columns (row, col).
    """
    height, width = plot_size

    if 0.7 < n_plots * width / height < 1.3:
        # Horizontal layout
        row = 1
        col = n_plots
    elif 0.7 < n_plots * height / width < 1.3:
        # Vertical layout
        row = n_plots
        col = 1
    else:
        # Matrix layout
        A = max(1, math.floor(math.sqrt(n_plots)))
        B = max(1, math.ceil(n_plots / A))
        if height > width * 0.7:
            row = A
            col = B
        else:
            row = B
            col = A

    return row, col


def nicelimit(x, scale=None, mode='hi'):
    if x == 0:
        return 1

    if scale is None:
        scale = 10 ** (math.ceil(math.log10(x)) - 1)

    if mode == 'lo':
        return scale * math.floor(x / scale)
    elif mode == 'hi':
        return scale * math.ceil(x / scale)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'lo' or 'hi'.")


class Logger(object):
    """
    This class allows saving console output to a file
    while also showing it in the terminal.
    """

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def close(self):
        self.terminal.close()
        self.log.close()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
