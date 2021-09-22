"""
Verify the functionality of the evaluation suite.

Executes the evaluation procedure against five samples and outputs the
results. Compare them with the results from the BSDS dataset to verify
that this Python port works properly.
"""

import os, argparse

import tqdm
from bsds.bsds_dataset import BSDSDataset
from bsds import evaluate_boundaries
from skimage.util import img_as_float
from skimage.io import imread

SAMPLE_NAMES = ['2018', '3063', '5096', '6046', '8068']
N_THRESHOLDS = 5

parser = argparse.ArgumentParser(description='Verify the BSDS-500 boundary '
                                             'evaluation suite')
parser.add_argument('bsds_path', type=str,
                    help='the root path of the BSDS-500 dataset')

args = parser.parse_args()

bsds_path = args.bsds_path
bench_dir_path = os.path.join(bsds_path, 'bench', 'data')

def load_gt_boundaries(sample_name):
    gt_path = os.path.join(bench_dir_path, 'groundTruth',
                           '{}.mat'.format(sample_name))
    return BSDSDataset.load_boundaries(gt_path)

def load_pred(sample_name):
    pred_path = os.path.join(bench_dir_path, 'png',
                             '{}.png'.format(sample_name))
    return img_as_float(imread(pred_path))


sample_results, threshold_results, overall_result = \
    evaluate_boundaries.pr_evaluation(N_THRESHOLDS, SAMPLE_NAMES,
                                      load_gt_boundaries, load_pred,
                                      progress=tqdm.tqdm)

print('Per image:')
for sample_index, res in enumerate(sample_results):
    print('{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        sample_index + 1, res.threshold, res.recall, res.precision, res.f1))


print('')
print('Overall:')
for thresh_i, res in enumerate(threshold_results):
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        res.threshold, res.recall, res.precision, res.f1))

print('')
print('Summary:')
print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'
      '{:<10.6f}'.format(
    overall_result.threshold, overall_result.recall,
    overall_result.precision, overall_result.f1, overall_result.best_recall,
    overall_result.best_precision, overall_result.best_f1,
    overall_result.area_pr)
)
