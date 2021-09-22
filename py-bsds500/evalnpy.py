
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('run', 'setup.py build_ext --inplace')


# # In[2]:


import numpy as np

"""
Verify the functionality of the evaluation suite.

Executes the evaluation procedure against five samples and outputs the
results. Compare them with the results from the BSDS dataset to verify
that this Python port works properly.
"""

import os, argparse

import tqdm
from bsds import evaluate_boundaries
from skimage.util import img_as_float
from skimage.io import imread
from scipy.io import loadmat
import cv2

# ---------------
# please fill in data_folder location and model name
# ---------------

SAMPLE_NAMES = [str(i) for i in np.arange(141)] 
N_THRESHOLDS = 99
UPPER_BOUND = 800
LOWER_BOUND = 1300
print("UPPER_BOUND",UPPER_BOUND)
print("LOWER_BOUND",LOWER_BOUND)
scalefactor = 3
print("scalefactor", scalefactor)
model_name = "faultSeg3d_thebe" #********
print("model_name",model_name)

data_folder = "..<DATA FOLDER>.." #********
gtpath = '{}/processedThebe/test'.format(data_folder)
Predpath = './faultSeg/thebe_pred/{}'.format(model_name)
print(Predpath)

h,w = fault[0,UPPER_BOUND:LOWER_BOUND,:].shape

h_s, w_s = int(h/scalefactor), int(w/scalefactor)
print("h_s, w_s", h_s, w_s)

def load_gt_boundaries(sample_name):
    gt_path = os.path.join(gtpath,'{}.npy'.format(sample_name))
    gt = np.load(gt_path)
    gt = gt.astype(np.float32)
    gt = cv2.resize(gt[UPPER_BOUND:LOWER_BOUND,:],(w_s,h_s))
    gt = gt>0.5
    gt = gt.astype(np.float32)
    return gt

def load_pred(sample_name):
    pred_path = os.path.join(Predpath,'{}.npy'.format(sample_name))
    pred = np.load(pred_path)
    pred = cv2.resize(pred,(w_s,h_s))
    return pred


sample_results, threshold_results, overall_result =     evaluate_boundaries.pr_evaluation(N_THRESHOLDS, SAMPLE_NAMES,
                                      load_gt_boundaries, load_pred,
                                      progress=tqdm.tqdm, apply_thinning=False)

import pandas as pd
overallresults = np.zeros((N_THRESHOLDS,4))
print(overallresults.shape)

# In[4]:


print('Per image:')
print("res.threshold, res.recall, res.precision, res.f1")
for sample_index, res in enumerate(sample_results):
    print('{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        sample_index + 1, res.threshold, res.recall, res.precision, res.f1))

i=0
print('')
print('Overall:')
print("res.threshold, res.recall, res.precision, res.f1")
for thresh_i, res in enumerate(threshold_results):
    overallresults[i] = [res.threshold, res.recall, res.precision, res.f1]
    i+=1
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        res.threshold, res.recall, res.precision, res.f1))

print('')
print('Summary:')
print("threshold, recall, precision, f1, best_recall, best_precision, best_f1, area_pr")
print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'
      '{:<10.6f}'.format(
    overall_result.threshold, overall_result.recall,
    overall_result.precision, overall_result.f1, overall_result.best_recall,
    overall_result.best_precision, overall_result.best_f1,
    overall_result.area_pr)
)

# save_path = model_name + '-{:<0.6f}-.csv'.format(overall_result.area_pr) 
# print(save_path) 
# df = pd.DataFrame(overallresults)
# df.to_csv(save_path,header=["threshold", "recall", "precision", "f1"], index=False)