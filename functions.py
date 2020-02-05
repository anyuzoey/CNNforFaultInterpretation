
# coding: utf-8

# In[1]:


# https://www.kaggle.com/erikistre/pytorch-basic-u-net

import time
import cv2 #bgr order
import math
from itertools import compress,product
# from skimage.io import imread, imshow
# from skimage.transform import resize
# import skimage.io as io
# from skimage import data, color, io
from skimage import color # rgb order
from skimage.util.shape import view_as_windows
from typing import Tuple

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.switch_backend('agg')
import numpy as np  
import scipy.signal
import pandas as pd
import glob
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torchsummary import summary
from torch.autograd import Variable
# import lovasz_losses as L

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
from os import listdir
from os.path import isfile, join
from os import walk
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score

from albumentations import (
    HorizontalFlip, VerticalFlip, CLAHE,
    ShiftScaleRotate, OpticalDistortion, GridDistortion, ElasticTransform, HueSaturationValue,
    RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, IAASuperpixels

)


import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1 is visible to this code


class logMAEloss(nn.Module):
    def _init_(self):
#     def _init_(self,inputs,targets):
        super(logMAEloss,self)._init_()
#         self.inputs = inputs
#         self.targets = targets
        return
    
    def forward(self,inputs,targets):
        # ‑log((‑x)+1) 
        mae = torch.abs(inputs - targets)
        loss = -torch.log((-mae)+1.0)
        return torch.mean(loss)

# In[2]:
def cross_entropy_loss_HED(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.0 * num_positive / (num_positive + num_negative)
#     mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)

def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
#     mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)

def load_pathnames_filenames(path):
    for (_, dirnames, _) in walk(path):
        print("dirnames", dirnames)
        break

    label_path = join(path, dirnames[2])
    seismics_path = join(path, dirnames[1])
    seismic_labels_path = join(path, dirnames[0])

    for (_, _, filenames) in walk(label_path):
        labels = sorted(filenames)
    for (_, _, filenames) in walk(seismics_path):
        seismics = sorted(filenames)
    for (_, _, filenames) in walk(seismic_labels_path):
        seismic_labels = sorted(filenames)

    print("number of images in three folder: ",len(labels),len(seismics),len(seismic_labels))
    print("label_path: ", label_path)
    print("seismics_path: ", seismics_path)
    print("seismic_labels_path: ", seismic_labels_path)

    print(labels[1])
    print(seismics[1])
    print(seismic_labels[1])

    return label_path, seismics_path, seismic_labels_path, labels, seismics, seismic_labels



# idea from https://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
def apply_mask(img,true,predict,alpha): #RGB Color[1,0,0]
    rows, cols, channel = img.shape
    # Construct a colour image to superimpose
    
    true = np.squeeze(true)
    predict = np.squeeze(predict)
    color_mask = np.zeros((rows, cols, channel))
    color_mask[predict>0] = [1,0,0]
    color_mask[true>0]= [0,0,1]
    color_mask[np.bitwise_and((true>0),(predict>0))]= [0,1,0]
    

#     # Construct RGB version of grey-level image
#     img_color = np.dstack((img, img, img))
    img_color = img

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    return img_masked
    

# def resizeImg(img, isMask,im_height,im_width):
#     if isMask==True:
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
#         img = np.invert(img)
#     else:
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)        
# #     print(img.shape,"-----1")
#     if img.shape[0:1]!=(1405,865):
#         img = cv2.resize(img, (1405, 865)) #,interpolation=cv2.INTER_NEAREST
# #     print(img.shape,"-----2")
#     img = img[120:120+im_height, 79:79+im_width]
# #     print(img.shape,"-----3")
# #     img = np.expand_dims(img, axis=-1)
    
#     return img

def resizeImg(img, isMask,im_height,im_width):
    if isMask==True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.invert(img)
#     else:
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)        
#     print(img.shape,"-----1")
    if img.shape[0:1]!=(1405,865):
        img = cv2.resize(img, (1405, 865)) #,interpolation=cv2.INTER_NEAREST
#     print(img.shape,"-----2")
    img = img[120:120+im_height, 79:79+im_width]
#     print(img.shape,"-----3")
#     img = np.expand_dims(img, axis=-1)
    
    return img


def split_Image(bigImage,isMask,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number):
#     print(bigImage.shape)
    if isMask==True:
        arr = np.pad(bigImage,((top_pad,bottom_pad),(left_pad,right_pad)),"reflect")
        splits = view_as_windows(arr, (splitsize,splitsize),step=stepsize)
        splits = splits.reshape((vertical_splits_number*horizontal_splits_number,splitsize,splitsize))
    else: 
        arr = np.pad(bigImage,((top_pad,bottom_pad),(left_pad,right_pad),(0,0)),"reflect")
        splits = view_as_windows(arr, (splitsize,splitsize,3),step=stepsize)
        splits = splits.reshape((vertical_splits_number*horizontal_splits_number,splitsize,splitsize,3))
    return splits # return list of arrays.


#idea from https://github.com/dovahcrow/patchify.py
def recover_Image(patches: np.ndarray, imsize: Tuple[int, int, int], left_pad,right_pad,top_pad,bottom_pad, overlapsize):
#     patches = np.squeeze(patches)
    assert len(patches.shape) == 5

    i_h, i_w, i_chan = imsize
    image = np.zeros((i_h+top_pad+bottom_pad, i_w+left_pad+right_pad, i_chan), dtype=patches.dtype)
    divisor = np.zeros((i_h+top_pad+bottom_pad, i_w+left_pad+right_pad, i_chan), dtype=patches.dtype)

#     print("i_h, i_w, i_chan",i_h, i_w, i_chan)
    n_h, n_w, p_h, p_w,_= patches.shape
    
    o_w = overlapsize
    o_h = overlapsize

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i,j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    recover = image / divisor
    return recover[top_pad:top_pad+i_h, left_pad:left_pad+i_w]

def recover_Image2(patches: np.ndarray, imsize: Tuple[int, int, int], left_pad,right_pad,top_pad,bottom_pad, overlapsize):
#     patches = np.squeeze(patches)
    assert len(patches.shape) == 5

    i_h, i_w, i_chan = imsize
    image = np.zeros((i_h+top_pad+bottom_pad, i_w+left_pad+right_pad, i_chan), dtype=patches.dtype)
    divisor = np.zeros((i_h+top_pad+bottom_pad, i_w+left_pad+right_pad, i_chan), dtype=patches.dtype)

#     print("i_h, i_w, i_chan",i_h, i_w, i_chan)
    n_h, n_w, p_h, p_w,_= patches.shape
    
    o_w = overlapsize
    o_h = overlapsize

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i,j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
#         divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    recover = image / 4
    return recover[top_pad:top_pad+i_h, left_pad:left_pad+i_w]

#https://github.com/Vooban/Smoothly-Blend-Image-Patches
def spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 3), 3)
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind

def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]
    
def strong_aug(p=1):
    return OneOf([
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
        IAASharpen(p=1),
        IAAEmboss(p=1),
        RandomBrightnessContrast(p=1),
        HorizontalFlip(p=1),
        VerticalFlip(p=1),
        Compose([VerticalFlip(p=1), HorizontalFlip(p=1)]),
        ElasticTransform(p=1, alpha=400, sigma=400 * 0.05, alpha_affine=400 * 0.03),
        GridDistortion(p=1),
        OpticalDistortion(p=1)
    ], p=p)
# def strong_aug(p=1):
#     return OneOf([
#         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
#         IAASharpen(p=1),
#         IAAEmboss(p=1),
#         RandomBrightnessContrast(p=1),
# #         HueSaturationValue(p=1),
#         HorizontalFlip(p=1),
#         VerticalFlip(p=1),
#         Compose([VerticalFlip(p=1), HorizontalFlip(p=1)]),
# #         CLAHE(clip_limit=2),
#         ElasticTransform(p=1, alpha=400, sigma=400 * 0.05, alpha_affine=400 * 0.03),
#         GridDistortion(p=1),
#         OpticalDistortion(p=1)
#     ], p=p)

#https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
SMOOTH = 1e-6
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0
#     print("=============outputs.sum((1,2))labels.sum=============")
#     print((outputs>0).sum((1,2)),(labels>0).sum((1,2)))
#     print("==========================")
#     print("intersection: ",intersection, "union: ",union)

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
#     print(iou)

#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

#     return thresholded  # Or thresholded.mean() if you are interested in average across the batch
    return iou

def iou_numpy(outputs: np.array, labels: np.array):
#     outputs = outputs.squeeze(1)
    intersection = (outputs & labels).sum((0,1))
    union = (outputs | labels).sum((0,1))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
#     thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return iou  # Or thresholded.mean()


# In[34]:


def acc_metrics(outputs, labels):
    TP=0
    TN=0
    FP=0
    FN=0
    # TP    predict 和 label 同时为1
    TP += ((outputs == 1) & (labels == 1)).sum()
    # TN    predict 和 label 同时为0
    TN += ((outputs == 0) & (labels == 0)).sum()
    # FN    predict 0 label 1
    FN += ((outputs == 0) & (labels == 1)).sum()
    # FP    predict 1 label 0
    FP += ((outputs == 1) & (labels == 0)).sum()
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    
    return p,r,F1,acc

# --------------------- 
# 作者：Link2Link 
# 来源：CSDN 
# 原文：https://blog.csdn.net/qq_15602569/article/details/79565402 
# 版权声明：本文为博主原创文章，转载请附上博文链接！







