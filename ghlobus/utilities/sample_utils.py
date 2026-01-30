"""
sample_utils.py

Various utilities related to sampling.

Author: Courosh Mehanian
        Daniel Shea
        Olivia Zahn

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import math
import torch
import random
import numpy as np
import pandas as pd

from typing import List, Union


def expand_list(arr: list, target_count: int, max_replicates: int = 10) -> list:
    """
    Expands a list to achieve target list length by replication.
    Assumes that target_count > len(arr).

    Parameters
    ----------
        arr: list             List to replicate elements of.
        target_count: int     Target length of the list.
        max_replicates: int   Maximum factor to replicated

    Returns
    -------
        list:                 List with replicated elements to achieve target length.
    """
    # get length of list
    actual_count = len(arr)
    # Compute ideal values
    whole_factor = target_count // actual_count
    remnant_count = target_count % actual_count
    # Check if we need to clip whole_factor
    if whole_factor > max_replicates:
        whole_factor = max_replicates
        remnant_count = 0
    # Replicate
    arr_rep = arr * whole_factor
    arr_rep.extend(arr[:remnant_count])

    return arr_rep


def subsample_frames(frames: torch.Tensor, k: int = 50) -> torch.Tensor:
    """
    Subsample `frames` Tensor, which is assumed to have dimensions (n,...) where
    n=number of samples that can be selected from.

    Parameters:
        frames: torch.Tensor        Frames Tensor to subsample
        k: int                      Number of frames to take from the frames Tensor

    Returns:
        frames: torch.Tensor        Subsampled frames
    """
    n = frames.shape[0]
    all_idcs = list(range(0, n))
    # Select only 'k' frames
    idcs = random.choices(all_idcs, k=k)
    # Sort them, so they appear in order.
    idcs = sorted(idcs)
    # Return the subset of frames
    return frames[idcs, ...]


def inference_subsample(frames: torch.Tensor, k: int = 50) -> torch.Tensor:
    """
    Consistent k-frames subsampling for inference.

    Parameters:
        frames: torch.Tensor        Frames Tensor to subsample
        k: int                      Number of frames to take from the frames Tensor

    Returns:
        frames: torch.Tensor        Subsampled frames
    """
    total_frames = frames.shape[0]
    if k >= total_frames:
        return frames

    skip_value = math.floor(total_frames/k)
    new_indcs = list(range(0, total_frames, skip_value))[0:k]
    frames = frames[new_indcs]
    return frames


def random_jitter_subsample(n: int, k: int, random_step: int = random.randint(1, 5)) -> list:
    """
    Randomly subsamples k indices from an array of length n.

    Args:
        n: int                 Length of the array.
        k: int                 Number of indices to subsample.
        random_step: int       Random step size for jittering the indices (optional).
                               Defaults to a random value between 1 and 5.

    Returns:
        torch.Tensor:          Subsampled tensor of frames.
    """
    if k >= n:
        indices = subsample_frames(np.array(range(n), dtype=int), k)
    else:
        if k * random_step < n:
            random_start = random.randint(0, n - k * random_step)
            indices = np.array(range(random_start, n, random_step)[0:k])
        else:
            random_step = random_step - 1
            indices = random_jitter_subsample(n, k, random_step)

    return indices


def random_jitter_subsample_frames(frames: torch.Tensor,  k: int) -> torch.Tensor:
    """
    Randomly subsamples k frames from a tensor of frames with random jitter.

    Args:
        frames: torch.Tensor   input tensor of frames.
        k: int                 number of frames to subsample.

    Returns:
        torch.Tensor:          subsampled tensor of frames.
    """
    # get the number of frames
    n = frames.shape[0]
    # compute the random jitter  indices
    inds = random_jitter_subsample(n, k)
    # grab frames and return
    return frames[inds, ...]


def uniformly_distributed_subsample(n: int, k: int) -> list:
    """
    Subsamples frames from a sequence to achieve a uniform distribution.

    This function selects `k` frames from a total of `n` frames. 
    The selection aims to space the frames evenly throughout the original sequence.

    Args:
        n:   The total number of frames.
        k:   The number of frames to select.

    Returns:
        A list of integer indices representing the selected frames.
    """
    # More requested than available, provide first k
    if k >= n:
        indices = list(range(n))[:k]
    else:
        # Less requested than available, sample uniformly, rounding to integers
        skip_value = float(n)/float(k)
        indices = np.round(np.arange(0, n, skip_value, dtype=float)).astype(int)
        indices = indices[:k].tolist()

    return indices


def uniformly_distributed_subsample_frames(frames: torch.Tensor, k: int) -> torch.Tensor:
    """
    Subsamples frames uniformly.

    Args:
        frames: torch.Tensor    input video sequence
        k: int                  number of frames to retain
    Returns:
        frames: torch.Tensor    k subsampled frames
    """
    # get the number of frames
    n = frames.shape[0]
    # compute the matern process indices
    inds = uniformly_distributed_subsample(n, k)

    # grab frames and return
    return frames[inds, ...]


def matern_subsample(n: int, k: int, r: int = 4, factor: float = 4) -> list:
    """
    Samples from N items according to Matern hard-core process
    which ensures that items are not too close to each other
    Args:
        n: int           number of items
        k: int           number of items to retain
        r: int           hard core radius
        factor: float    generate factor * k initial
                             Poisson random indices, then thin to k
    Returns:
        indcs: list      k subsampled indices as list
    """

    #set seed
    random.seed(1234)
    np.random.seed(2345)

    # determine how many in base Poisson process
    n_base = min(int(math.ceil(factor * k)), n)
    # determine maximum value of r
    r = min(r, int(math.floor(n/k)))
    # set of indices to choose from
    all_inds = list(range(0 - r, n + r))
    # check for insufficiency
    if k == n:
        return all_inds
    elif k > n:
        raise ValueError("Cannot use Matern when number of samples exceeds elements")
    # generate base process
    inds = np.array(sorted(random.sample(all_inds, k=n_base)))

    # generate random marks for each point
    mark = np.random.rand(n_base)
    keep = np.zeros(n_base, dtype=bool)

    # iterate through indices and thin according to distance
    for i, ind in enumerate(inds):
        # distance to this index
        dist = np.abs(inds - ind)
        # mask of points within Matern hard core distance of ind
        clash = dist < r
        if np.any(clash):
            clash_i = np.flatnonzero(clash)
            # mark points for deletion
            clash_age = mark[clash]
            # keep "oldest" point
            keep_rel_i = np.argmax(clash_age)
            keep_abs_i = clash_i[keep_rel_i]
            keep[keep_abs_i] = True

    # remove indices outside 0 ... n-1 and those not to be kept
    inds = inds[np.logical_and(np.logical_and(
        inds >= 0, inds < n), keep)].tolist()

    # check if we have more or less than requested
    if len(inds) > k:
        # if more than k, subsample
        inds = random.sample(inds, k=k)
        inds.sort()
    elif len(inds) < k:
        # if less, subsample with replacement
        inds = random.choices(inds, k=k)
        inds.sort()

    return inds


def matern_subsample_frames(frames: torch.Tensor, k: int, r: int = 4, factor: float = 4) -> torch.Tensor:
    """
    Subsamples frames according to Matern hard-core process
    which ensures that items are not too close to each other
    Args:
        frames: torch.Tensor    input video sequence
        k: int                  number of frames to retain
        r: int                  hard core radius
        factor: float           generate factor * k initial
                                    Poisson random indices, then thin to k
    Returns:
        frames: torch.Tensor    k subsampled frames
    """
    # get the number of frames
    n = frames.shape[0]
    # compute the matern process indices
    inds = matern_subsample(n, k, r=r, factor=factor)

    # grab frames and return
    return frames[inds, ...]


def half_subsample(n: int, k: int) -> list:
    """
    Samples from N items according to 1/2 sampling process.

    Args:
        n: int          number of frames
        k: int          number of frames to retain
    Returns:
        indcs: list     k subsampled indices as list
    """
    # determine number of output clips
    n_clips = max(1, math.ceil(n / (2 * k)))
    # container for frame_inds
    frame_inds = [[] for _ in range(n_clips)]
    for clip_ind in range(n_clips):
        # initial stab at 1/2 sampling
        beg_ind = clip_ind * 2 * k
        end_ind = min(beg_ind + 2 * k, n - (clip_ind % 2))
        # generate 1/2 samples, but displace odd ones by 1
        fm_inds = np.arange(beg_ind, end_ind, 2) + (clip_ind % 2)
        # how many indices are there
        count = len(fm_inds)
        shortfall = k - count
        # check if too few
        if shortfall > 0 and clip_ind > 0:
            # sample from previous clip, even or odd indices
            beg_ind = (clip_ind - 1) * 2 * k
            end_ind = beg_ind + 2 * k - (clip_ind % 2)
            supp_inds = np.arange(beg_ind, end_ind, 2) + (clip_ind % 2)
            # take indices on the end
            supp_inds = supp_inds[-shortfall:]
            # supplement indices for clip_ind from clip_ind -1
            frame_inds[clip_ind] = sorted(list(fm_inds) + list(supp_inds))
        elif shortfall > 0 and clip_ind == 0:
            # sample from same clip, using odd indices
            beg_ind = 0
            end_ind = n - 1
            supp_inds = np.arange(beg_ind, end_ind, 2) + 1
            # take indices on the end
            supp_inds = supp_inds[-shortfall:]
            # supplement indices for 0 even with 0 odd
            frame_inds[clip_ind] = sorted(list(fm_inds) + list(supp_inds))
        else:
            # no need to supplement
            frame_inds[clip_ind] = list(fm_inds)

    return frame_inds


def make_clips(video_frame_indices: List[List[int]], n_frames: int) -> List[List[int]]:
    """
    Splices a set of videos (e.g., an exam) into a collection of (mostly)
    non-overlapping clips, each of which is half-sampled from the parent video.

    Parameters:
        video_frame_indices: List[List[int]]  - List of lists of frame indices from a group
                                                of videos (e.g., from and exam)
        n_frames: int                         - Desired number of frames in each clip

    Returns
        clip_indices: List[int]               - List of clip indices (absolute relative
                                                to the entire set of frames from the original
                                                set of videos
    """
    # clip indices container
    clip_indices = list()

    # Initialize frame counter that determines global exam-wide indices
    previous_video_end = 0
    # Go through each video splicing clips at half-sampling rate
    for frame_indices in video_frame_indices:
        # Get the length of current video
        n = len(frame_indices)
        # Get clip indices at half-sampling rate for this video
        this_video_clip_indices = half_subsample(n, n_frames)
        # Adjust the indices to make them global across the exam
        this_video_clip_indices = [np.array(x, dtype=int) + previous_video_end for x in this_video_clip_indices]
        # Add the clips from the current video to the accumulated list
        clip_indices.extend(this_video_clip_indices)
        # Increment the frame counter to track global exam indices
        previous_video_end = previous_video_end + n

    return clip_indices


def construct_bins(ftr, nbins, balance):
    """
    This function constructs digitization bins for a feature vector.

    It finds quantiles of the original distribution of a feature,
    say GA, for example. There are two extremes. At the one extreme, the
    quantiles are evenly spaced, e.g., [0, 0.2, 0.4, 0.6, 0.8, 1.0].
    If you digitize GA according to these quantiles, and then balance
    the distribution of the resulting digitized GAs, you will replicate
    the original distribution. At the other extreme, you can define
    regularly spaced GA intervals, e.g., [60, 104, 148, 192, 236, 280],
    and then compute the quantiles of the data at these GA values. Suppose
    the quantiles are: [0, 0.04, 0.18, 0.43, 0.79, 1.0]. If you digitize
    according to these quantiles and balance, you will get an even
    distribution of digitized GA. Or you can follow an approach where
    you computed a weighted combination of the two quantile vectors.
    With balance=0, you replicate original distribution, with balance=1,
    you get a perfectly balanced distribution. With balance somewhere
    between [0, 1] the final distribution will land somewhere between
    balanced and the original.

    Parameters:
        ftr         feature that is to be digitized
        nbins       number of bins to digitize into
        balance     number in [0, 1]
                        balance = 0 means stratified (same as original
                                    distribution)
                        balance = 1 means balanced (all feature values
                                    equally populated)
    Returns:
        bins        vector of feature quantization bins
    """
    # compute evenly spaced ftr bins from min to max + epsilon
    even_ftr_bins = np.linspace(ftr.min(), ftr.max(), nbins + 1)
    # compute quantiles of even feature bins
    even_ftr_dqs = np.empty(nbins)
    for i in range(1, nbins + 1):
        even_ftr_dqs[i - 1] = \
            np.logical_and(ftr > even_ftr_bins[i - 1],
                           ftr <= even_ftr_bins[i]).sum() / len(ftr)
    # split quantiles evenly
    even_qnt_dqs = np.ones(nbins) / nbins
    # compute combination of even feature bins dqs and even quantile dqs
    dqs = balance * even_ftr_dqs + (1 - balance) * even_qnt_dqs
    # compute quantiles
    q_vals = compute_q_vals(dqs)
    # compute combined feature bins
    bins = np.quantile(ftr, q_vals)
    bins[-1] = 1.0000001 * bins[-1]

    return bins


def bin_discrete_feature(ftr, nbins):
    """
    This function bins a discrete feature according to the bins provided.

    Parameters:
        ftr:         Feature to digitize
        nbins:       Number of bins to group feature into
    Returns:
        y:           Binned feature
    """
    # create output array
    y = - np.ones(len(ftr), dtype=int)

    # bin the feature values
    for ind, vals in enumerate(nbins):
        y[np.in1d(ftr, vals)] = ind

    return y


def balanced_sample(y: np.ndarray,
                    ftr: Union[np.ndarray, None] = None,
                    fraction: float = 1,
                    balance: float = 0,
                    strategy: str = 'combo') -> np.ndarray:
    """
    This function generates sampled indices (fraction of total counts)
    that balance the counts of each unique value in the digitized feature y

    Parameters:
        y          Digitized feature or discrete category
        ftr        Secondary feature (e.g., quantized from continuous)
        fraction   Number in [0, 1] fraction of # original instances to sample
        balance    Number in [0, 1] spectrum between stratified and balanced
                   balance = 0 is stratified
                   balance = 1 is completely balanced
                   balance = 0.5 is halfway between stratified and balanced
                   choose balance = 1 for continuous features because method of
                   bin construction has already dialed in the required mix in
                   the spectrum from stratified to balanced.
                   recommended to use with strategy = 'combo' to get correct final count
        strategy   Strategy to use for balancing the data; the choices are:
                   'over'   oversampling only, total will exceed fraction requested
                   'under'  undersampling only, total will far short of fraction requested
                   'combo'  over- and under-sampling, total will equal fraction requested
    Returns:
        balanced_inds  Indices that define the sample
    """
    # find counts of all inputs
    unique, counts = np.unique(y, return_counts=True)
    # find number of bins
    nbins = len(unique)

    # check if we are pre-balancing across a continuous feature
    if ftr is not None:
        # balance across 2nd feature for each unique y
        ftr2_balanced_inds = np.empty(0, dtype=int)
        for idx, u in enumerate(unique):
            # indices for this value of y
            mask = np.in1d(y, u)
            inds = np.where(mask)[0]
            # 2nd feature at those indices
            ftr_u = ftr[mask]
            # balance on the 2nd feature to get relative indices
            rel_inds = balanced_sample(ftr_u,
                                       ftr=None,
                                       fraction=0.9,
                                       balance=1,
                                       strategy='combo')
            # get absolute indices
            abs_inds = inds[rel_inds]
            ftr2_balanced_inds = np.append(ftr2_balanced_inds, abs_inds)
        # balance y over the provided feature
        yb = y[ftr2_balanced_inds]
        # redo counts after feature balancing
        _, counts = np.unique(yb, return_counts=True)
    else:
        # no 2nd feature
        yb = y
        ftr2_balanced_inds = None

    # find target count for each bin of y
    # these counts correspond to stratified sampling
    f_counts = np.array([round(fraction * x) for x in counts])
    # choose target count for each bin based on strategy
    if strategy == 'combo':
        # nominal target counts for each bin, meets total target count
        t_counts = [round(f_counts.sum() / nbins)] * nbins
    elif strategy == 'over':
        # oversampling: choose majority class count for each bin
        t_counts = [f_counts.max()] * nbins
    elif strategy == 'under':
        # undersampling: choose minority class count for each bin,
        # but no more than what target count demands  (combo)
        t_counts = [min(counts.min(), round(f_counts.sum() / nbins))] * nbins
    else:
        # invalid strategy
        raise ValueError(f"Invalid strategy: {strategy}")
    # choose final target counts based on balance parameter
    target_counts = [round(balance * t + (1 - balance) * f) for t, f in zip(t_counts, f_counts)]
    # randomly sample indices for each y bin
    inds = [[] for _ in range(nbins)]
    for idx, u in enumerate(unique):
        inds[idx] = list(np.flatnonzero(yb == u))
    # for each unique value
    for idx, count in enumerate(counts):
        # indices for unique value u
        base = inds[idx]
        target = target_counts[idx]
        if target <= count:
            # enough samples in class u; undersample
            inds[idx] = random.sample(base, k=target)
        else:
            # more samples needed than available; oversample
            # even multiples of indices for this element
            multiple = target // count
            # remainder of indices for this element
            remainder = target % count
            inds[idx] = base * multiple
            inds[idx].extend(random.sample(base, k=remainder))

    # flatten the list of indices
    balanced_inds = np.array([x for sublist in inds for x in sublist])

    # check if we need to map balanced_inds back to original indices
    if ftr is not None:
        balanced_inds = ftr2_balanced_inds[balanced_inds]

    # shuffle the indices
    np.random.shuffle(balanced_inds)

    return balanced_inds


def compute_q_vals(dq):
    """
    This function computes the cumulative quantile vector
    from a vector of differential quantiles. It prepends
    zero to ensure the entire [0, 1] interval is covered.
    """
    return np.concatenate(([0], dq.cumsum()))


def compute_equal_sampling_weights(categorical_column: pd.Series) -> list:
    """
    Given a categorical column, this computes a list of weights by index
    such that each category has an equal chance of showing up in the sampled
    indices by a sampler.

    Parameters
        categorical_column: pd.Series   Column with the categorical values

    Returns
    weights_by_index: list              List of same length as categorical column with
                                        weights yielding roughly even sampling by category
    """
    vcs = categorical_column.value_counts(dropna=False)
    # Total count of entries in the categorical column
    n = vcs.sum()
    # Compute the weights
    # (desired sampling ratio (even)) / (ratio in samples)
    weights = (1/len(vcs.index)) / (vcs/n)
    # Now compute the weights for each index, based on the categorical column
    weights_by_index = categorical_column.apply(lambda x: weights[x])
    weights_by_index = weights_by_index.tolist()
    # Return the weights list
    return weights_by_index


def compute_equal_sampling_weights_by_trimester(df: pd.DataFrame) -> list:
    """
    Compute equal sampling weights by quantizing gestational age into trimester bins.
    
    This function divides gestational age (ga_boe) values into three trimester periods
    based on standard pregnancy trimesters and computes sampling weights that ensure 
    equal representation across all trimester bins when sampling from the dataset.
    
    Parameters:
        df: pd.DataFrame  DataFrame containing 'ga_boe' column with gestational
                            age values
        
    Returns:
        list:             List of sampling weights corresponding to each row in
                            the DataFrame, where weights are computed to achieve
                            equal sampling across trimester bins

    Note:
        The trimester bins are:
        - Trimester 1: 0-98 days
        - Trimester 2: 98-196 days
        - Trimester 3: 196-300 days
    """
    assert ('ga_boe' in df.columns)
    trimester = pd.cut(
        x=df['ga_boe'].astype(float),
        bins=(0, 98, 196, 300),
        labels=(1, 2, 3),
    )
    # Return the weights list
    return compute_equal_sampling_weights(trimester)


def compute_equal_sampling_weights_by_month(df: pd.DataFrame) -> list:
    """
    Compute equal sampling weights by quantizing gestational age into monthly bins.
    
    This function divides gestational age (ga_boe) values into monthly intervals
    (30-day periods) and computes sampling weights that ensure equal representation
    across all monthly bins when sampling from the dataset.
    
    Parameters:
        df: pd.DataFrame  DataFrame containing 'ga_boe' column with gestational
                            age values
        
    Returns:
        list:             List of sampling weights corresponding to each row in
                           the DataFrame, where weights are computed to achieve
                           equal sampling across monthly bins
                           
    Raises:
        AssertionError:   If 'ga_boe' column is not present in the DataFrame
    """
    assert ('ga_boe' in df.columns)
    month = df['ga_boe'].astype(float) // 30
    # Return the weights list
    return compute_equal_sampling_weights(month)


def compute_equal_sampling_weights_by_week(df: pd.DataFrame) -> list:
    """
    Compute equal sampling weights by quantizing gestational age into weekly bins.
    
    This function divides gestational age (ga_boe) values into weekly intervals
    (7-day periods) and computes sampling weights that ensure equal representation
    across all weekly bins when sampling from the dataset.
    
    Parameters:
        df: pd.DataFrame    DataFrame containing 'ga_boe' column with gestational age values
        
    Returns:
        list:               List of sampling weights corresponding to each row in the DataFrame,
                           where weights are computed to achieve equal sampling across weekly bins
                           
    Raises:
        AssertionError:     If 'ga_boe' column is not present in the DataFrame
    """
    assert ('ga_boe' in df.columns)
    week = df['ga_boe'].astype(float) // 7
    # Return the weights list
    return compute_equal_sampling_weights(week)


def compute_equal_sampling_weights_by_efw(df: pd.DataFrame) -> list:
    """
    Compute equal sampling weights by quantizing effective fetal weight into staged bins.
    
    This function divides estimated fetal weight (efw_boe) values into three predefined
    weight categories (0-1500g, 1500-3000g, 3000-5000g) and computes sampling weights
    that ensure equal representation across all weight stage bins when sampling from the dataset.
    
    Parameters:
        df: pd.DataFrame  DataFrame containing 'efw_boe' column with estimated
                            fetal weight values
        
    Returns:
        list:             List of sampling weights corresponding to each row in the
                           DataFrame, where weights are computed to achieve equal
                           sampling across EFW stage bins
                           
    Raises:
        AssertionError:     If 'efw_boe' column is not present in the DataFrame
        
    Note:
        The weight bins are:
        - Stage 1: 0-1500 grams
        - Stage 2: 1500-3000 grams  
        - Stage 3: 3000-5000 grams
    """
    assert ('efw_boe' in df.columns)
    trimester = pd.cut(
        x=df['efw_boe'].astype(float),
        bins=(0, 1500, 3000, 5000),
        labels=(1, 2, 3),
    )
    # Return the weights list
    return compute_equal_sampling_weights(trimester)


def evenly_spaced_elements(arr, n):
    """
    Get n evenly spaced elements from an array.

    Args:
        arr (List):    Array to get elements from.
        n (int):       Number of elements to get.

    Returns:
        List:          n evenly spaced elements
    """
    length = len(arr)
    if n > length:
        raise ValueError("n should be less than or equal to the length of the array")
    step = (length - 1) / (n - 1)
    result = [arr[int(round(i * step))] for i in range(n)]
    return result
