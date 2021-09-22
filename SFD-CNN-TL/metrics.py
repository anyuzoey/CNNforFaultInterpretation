#!/usr/bin/env python
# Copyright 2019 Augusto Cunha and Axelle Pochet
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this code and 
# associated documentation files, to deal in the code without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the code, and to permit persons to whom the code is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the code.
#
# THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE CODE OR THE USE OR OTHER DEALINGS IN THE CODE.
__license__ = "MIT"
__author__ = "Augusto Cunha, Axelle Pochet"
__email__ = "acunha@tecgraf.puc-rio.br, axelle@tecgraf.puc-rio.br"
__credits__ = ["Augusto Cunha", "Axelle Pochet", "Helio Lopes", "Marcelo Gattass"]

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def generate_metrics(classesPredictionList, classesProbaPredictionList, y, verbose=True):
    """
    Metrics evaluate function:

    Compute all related metrics
    """
    # Count True Positive, True Negative, False Positive, False Negative
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range( len(y) ):
        prediction = classesPredictionList[i]
        expected = 0 if y[i][0] == 1 else 1 # Verifiy in ground truth [1 0] witch class is
        if(prediction == expected):
            if(expected == 1): #Fault
                TP = TP + 1
            else: #NonFault
                TN = TN + 1
        else:
            if(expected == 1): #Fault
                FN = FN + 1
            else: #NonFault
                FP = FP + 1

    sensitivity = safe_div(TP , TP + FN)
    specificity = safe_div(TN , TN + FP) 
    accuracy = safe_div(TP + TN , TP + TN + FP + FN)
    precision = safe_div(TP , TP + FP)
    recall = sensitivity
    F1_score = safe_div(2 * (precision * recall) , precision + recall)
    
    if(verbose):
        print("METRICS:")
        print("Sensitivity:",sensitivity)
        print("Specificity:",specificity)
        print("Accuracy:",accuracy)
        print("Precision:",precision)
        print("Recall:",recall)
        print("F1 Score:",F1_score)
    
    ########### ROC, AUC #############
    # compute ROC, AUC
    fpr, tpr, thresholds = roc_curve(np.argmax(y,1), classesProbaPredictionList[:,1])
    AUC = auc(fpr, tpr)
    
    if(verbose):
        print("AUC:",AUC)
    
    return sensitivity, specificity, accuracy, precision, recall, F1_score, AUC