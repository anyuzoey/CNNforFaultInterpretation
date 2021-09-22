from collections import namedtuple
import numpy as np
from . import thin, correspond_pixels


def evaluate_boundaries_bin(predicted_boundaries_bin, gt_boundaries,
                            max_dist=0.0075, apply_thinning=False):
    """
    Evaluate the accuracy of a predicted boundary.

    :param predicted_boundaries_bin: the predicted boundaries as a (H,W)
    binary array
    :param gt_boundaries: a list of ground truth boundaries, as returned
    by the `load_boundaries` or `boundaries` methods
    :param max_dist: (default=0.0075) maximum distance parameter
    used for determining pixel matches. This value is multiplied by the
    length of the diagonal of the image to get the threshold used
    for matching pixels.
    :param apply_thinning: (default=True) if True, apply morphologial
    thinning to the predicted boundaries before evaluation
    :return: tuple `(count_r, sum_r, count_p, sum_p)` where each of
    the four entries are float values that can be used to compute
    recall and precision with:
    ```
    recall = count_r / (sum_r + (sum_r == 0))
    precision = count_p / (sum_p + (sum_p == 0))
    ```
    """
    acc_prec = np.zeros(predicted_boundaries_bin.shape, dtype=bool)
    predicted_boundaries_bin = predicted_boundaries_bin != 0

    if apply_thinning:
        predicted_boundaries_bin = thin.binary_thin(predicted_boundaries_bin)

    sum_r = 0
    count_r = 0
    for gt in gt_boundaries:
        match1, match2, cost, oc = correspond_pixels.correspond_pixels(
            predicted_boundaries_bin, gt, max_dist=max_dist
        )
        match1 = match1 > 0
        match2 = match2 > 0
        # Precision accumulator
        acc_prec = acc_prec | match1
        # Recall
        sum_r += gt.sum()
        count_r += match2.sum()

    # Precision
    sum_p = predicted_boundaries_bin.sum()
    count_p = acc_prec.sum()

    return count_r, sum_r, count_p, sum_p


def evaluate_boundaries(predicted_boundaries, gt_boundaries,
                        thresholds=99, max_dist=0.0075, apply_thinning=False,
                        progress=None):
    """
    Evaluate the accuracy of a predicted boundary and a range of thresholds

    :param predicted_boundaries: the predicted boundaries as a (H,W)
    floating point array where each pixel represents the strength of the
    predicted boundary
    :param gt_boundaries: a list of ground truth boundaries, as returned
    by the `load_boundaries` or `boundaries` methods
    :param thresholds: either an integer specifying the number of thresholds
    to use or a 1D array specifying the thresholds
    :param max_dist: (default=0.0075) maximum distance parameter
    used for determining pixel matches. This value is multiplied by the
    length of the diagonal of the image to get the threshold used
    for matching pixels.
    :param apply_thinning: (default=True) if True, apply morphologial
    thinning to the predicted boundaries before evaluation
    :param progress: a function that can be used to monitor progress;
    use `tqdm.tqdm` or `tdqm.tqdm_notebook` from the `tqdm` package
    to generate a progress bar.
    :return: tuple `(count_r, sum_r, count_p, sum_p, thresholds)` where each
    of the first four entries are arrays that can be used to compute
    recall and precision at each threshold with:
    ```
    recall = count_r / (sum_r + (sum_r == 0))
    precision = count_p / (sum_p + (sum_p == 0))
    ```
    The thresholds are also returned.
    """
    if progress is None:
        progress = lambda x, *args, **kwargs: x

    # Handle thresholds
    if isinstance(thresholds, int):
        thresholds = np.linspace(1.0 / (thresholds + 1),
                                 1.0 - 1.0 / (thresholds + 1), thresholds)
    elif isinstance(thresholds, np.ndarray):
        if thresholds.ndim != 1:
            raise ValueError('thresholds array should have 1 dimension, '
                             'not {}'.format(thresholds.ndim))
        pass
    else:
        raise ValueError('thresholds should be an int or a NumPy array, not '
                         'a {}'.format(type(thresholds)))

    sum_p = np.zeros(thresholds.shape)
    count_p = np.zeros(thresholds.shape)
    sum_r = np.zeros(thresholds.shape)
    count_r = np.zeros(thresholds.shape)

    for i_t, thresh in enumerate(progress(list(thresholds))):
        predicted_boundaries_bin = predicted_boundaries >= thresh

        acc_prec = np.zeros(predicted_boundaries_bin.shape, dtype=bool)

        if apply_thinning:
            predicted_boundaries_bin = thin.binary_thin(
                predicted_boundaries_bin)

#         for gt in gt_boundaries:
        gt = gt_boundaries
#         print(predicted_boundaries_bin.shape,gt.shape)

        match1, match2, cost, oc = correspond_pixels.correspond_pixels(
            predicted_boundaries_bin, gt, max_dist=max_dist
        )
        match1 = match1 > 0
        match2 = match2 > 0
        # Precision accumulator
        acc_prec = acc_prec | match1
        # Recall
        sum_r[i_t] = gt.sum()
        count_r[i_t] = match2.sum()

        # Precision
        sum_p[i_t] = predicted_boundaries_bin.sum()
        count_p[i_t] = acc_prec.sum()

    return count_r, sum_r, count_p, sum_p, thresholds



def compute_rec_prec_f1(count_r, sum_r, count_p, sum_p):
    """
    Computer recall, precision and F1-score given `count_r`, `sum_r`,
    `count_p` and `sum_p`; see `evaluate_boundaries`.
    :param count_r:
    :param sum_r:
    :param count_p:
    :param sum_p:
    :return: tuple `(recall, precision, f1)`
    """
    rec = count_r / (sum_r + (sum_r == 0))
    prec = count_p / (sum_p + (sum_p == 0))
    f1_denom = (prec + rec + ((prec+rec) == 0))
    f1 = 2.0 * prec * rec / f1_denom
    return rec, prec, f1


SampleResult = namedtuple('SampleResult', ['sample_name', 'threshold',
                                           'recall', 'precision', 'f1'])
ThresholdResult = namedtuple('ThresholdResult', ['threshold', 'recall',
                                                 'precision', 'f1'])
OverallResult = namedtuple('OverallResult', ['threshold', 'recall',
                                             'precision', 'f1',
                                             'best_recall', 'best_precision',
                                             'best_f1', 'area_pr'])

def pr_evaluation(thresholds, sample_names, load_gt_boundaries, load_pred,
                  progress=None, apply_thinning = False):
    """
    Perform an evaluation of predictions against ground truths for an image
    set over a given set of thresholds.

    :param thresholds: either an integer specifying the number of thresholds
    to use or a 1D array specifying the thresholds
    :param sample_names: the names of the samples that are to be evaluated
    :param load_gt_boundaries: a callable that loads the ground truth for a
        named sample; of the form `load_gt_boundaries(sample_name) -> gt`
        where `gt` is a 2D NumPy array
    :param load_pred: a callable that loads the prediction for a
        named sample; of the form `load_gt_boundaries(sample_name) -> gt`
        where `gt` is a 2D NumPy array
    :param progress: default=None a callable -- such as `tqdm` -- that
        accepts an iterator over the sample names in order to track progress
    :return: `(sample_results, threshold_results, overall_result)`
    where `sample_results` is a list of `SampleResult` named tuples with one
    for each sample, `threshold_results` is a list of `ThresholdResult`
    named tuples, with one for each threshold and `overall_result`
    is an `OverallResult` named tuple giving the over all results. The
    attributes in these structures will now be described:

    `SampleResult`:
    - `sample_name`: the name identifying the sample to which this result
        applies
    - `threshold`: the threshold at which the best F1-score was obtained for
        the given sample
    - `recall`: the recall score obtained at the best threshold
    - `precision`: the precision score obtained at the best threshold
    - `f1`: the F1-score obtained at the best threshold\

    `ThresholdResult`:
    - `threshold`: the threshold value to which this result applies
    - `recall`: the average recall score for all samples
    - `precision`: the average precision score for all samples
    - `f1`: the average F1-score for all samples

    `OverallResult`:
    - `threshold`: the threshold at which the best average F1-score over
        all samples is obtained
    - `recall`: the average recall score for all samples at `threshold`
    - `precision`: the average precision score for all samples at `threshold`
    - `f1`: the average F1-score for all samples at `threshold`
    - `best_recall`: the average recall score for all samples at the best
        threshold *for each individual sample*
    - `best_precision`: the average precision score for all samples at the
        best threshold *for each individual sample*
    - `best_f1`: the average F1-score for all samples at the best threshold
        *for each individual sample*
    - `area_pr`: the area under the precision-recall curve at `threshold`
    `
    """
    if progress is None:
        progress = lambda x, *args: x

    if isinstance(thresholds, int):
        n_thresh = thresholds
    else:
        n_thresh = thresholds.shape[0]

    count_r_overall = np.zeros((n_thresh,))
    sum_r_overall = np.zeros((n_thresh,))
    count_p_overall = np.zeros((n_thresh,))
    sum_p_overall = np.zeros((n_thresh,))

    count_r_best = 0
    sum_r_best = 0
    count_p_best = 0
    sum_p_best = 0

    sample_results = []
    for sample_index, sample_name in enumerate(progress(sample_names)):
        # Get the paths for the ground truth and predicted boundaries

        # Load them
        pred = load_pred(sample_name)
        gt_b = load_gt_boundaries(sample_name)

        # Evaluate predictions
        count_r, sum_r, count_p, sum_p, used_thresholds = \
            evaluate_boundaries(pred, gt_b, thresholds=thresholds,
                                apply_thinning = False)

        count_r_overall += count_r
        sum_r_overall += sum_r
        count_p_overall += count_p
        sum_p_overall += sum_p

        # Compute precision, recall and F1
        rec, prec, f1 = compute_rec_prec_f1(count_r, sum_r, count_p, sum_p)

        # Find best F1 score
        best_ndx = np.argmax(f1)

        count_r_best += count_r[best_ndx]
        sum_r_best += sum_r[best_ndx]
        count_p_best += count_p[best_ndx]
        sum_p_best += sum_p[best_ndx]

        sample_results.append(SampleResult(sample_name,
                                           used_thresholds[best_ndx],
                                           rec[best_ndx], prec[best_ndx],
                                           f1[best_ndx]))

    # Computer overall precision, recall and F1
    rec_overall, prec_overall, f1_overall = compute_rec_prec_f1(
        count_r_overall, sum_r_overall, count_p_overall, sum_p_overall)

    # Find best F1 score
    best_i_ovr = np.argmax(f1_overall)

    threshold_results = []
    for thresh_i in range(n_thresh):
        threshold_results.append(ThresholdResult(used_thresholds[thresh_i],
                                                 rec_overall[thresh_i],
                                                 prec_overall[thresh_i],
                                                 f1_overall[thresh_i]))


    rec_unique, rec_unique_ndx = np.unique(rec_overall, return_index=True)
    prec_unique = prec_overall[rec_unique_ndx]
    if rec_unique.shape[0] > 1:
        prec_interp = np.interp(np.arange(0, 1, 0.01), rec_unique,
                                prec_unique, left=0.0, right=0.0)
        area_pr = prec_interp.sum() * 0.01
    else:
        area_pr = 0.0

    rec_best, prec_best, f1_best = compute_rec_prec_f1(
        float(count_r_best), float(sum_r_best), float(count_p_best),
        float(sum_p_best)
    )

    overall_result = OverallResult(used_thresholds[best_i_ovr],
                                   rec_overall[best_i_ovr],
                                   prec_overall[best_i_ovr],
                                   f1_overall[best_i_ovr], rec_best,
                                   prec_best, f1_best, area_pr)

    return sample_results, threshold_results, overall_result
