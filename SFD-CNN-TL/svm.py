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

################# all imports #################
from __future__ import print_function
import numpy, os, time
import pandas as pd
from tensorflow import set_random_seed

numpy.random.seed(1337)
set_random_seed(1337)

from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import metrics
from sklearn.externals import joblib

def load_model(modelJsonPath, modelWeightsPath):
    ################# load base model #################
    jsonFile = open(modelJsonPath, 'r')
    loadedModelJson = jsonFile.read()
    jsonFile.close()
    base_model = model_from_json(loadedModelJson)
    base_model.load_weights(modelWeightsPath)


    # remove last layers
    for i in range (7):
        base_model.layers.pop()
        base_model.outputs = [base_model.layers[-1].output]

    # freeze layers
    for layer in base_model.layers[:7]:
        layer.trainable = False

    return base_model

def data(X_train, Y_train, numberOfClasses = 2):
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, random_state=1337)
    return x_train, y_train, x_test, y_test

def dataCV(trainFaultDirectory='dataset/fault/',trainNonFaultDirectory='dataset/nonfault/', modelJsonPath = 'base_model/model.json', modelWeightsPath = 'base_model/model.h5'):
    
    trainFaultURLList = os.listdir(trainFaultDirectory)
    trainNonFaultURLList = os.listdir(trainNonFaultDirectory)

    # read and save
    trainImageDataList = []
    trainClassesList = []
    for imageURL in trainFaultURLList:
        csv_file = trainFaultDirectory + imageURL
        df = pd.read_csv(csv_file, delimiter=' ', header = None)
        trainImageDataList.append(df.values)
        trainClassesList.append(1)
        
    for imageURL in trainNonFaultURLList:
        csv_file = trainNonFaultDirectory + imageURL
        df = pd.read_csv(csv_file, delimiter=' ', header = None)
        trainImageDataList.append(df.values)
        trainClassesList.append(0)
    
    # sparsify labels
    Y = trainClassesList

    # pass input as numpy arrays
    imageRows = 45
    imageCollumns = 45
    imageChannels = 1

    trainSamplesList = numpy.array( trainImageDataList) 
    trainSamplesList = trainSamplesList.reshape( trainSamplesList.shape[0], imageRows, imageCollumns, imageChannels )
    trainSamplesList = trainSamplesList.astype( 'float32' )
    
    X = trainSamplesList
    ## extract features as new input
    X = load_model(modelJsonPath, modelWeightsPath).predict(X)
    x_train = X
    y_train = Y
    x_test = []
    y_test = []
    
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test, numFolds= 5, c=1, k='linear', save = True, baseName='femlpModel'):
    """
    Model providing function:

    Create Keras model with SVM as classifier, compile test and generate metrics.
    """
    ################# define SVM #################
    clf = svm.SVC(kernel = k, C = c, probability=True, random_state=1337)
    clf.fit(x_train, y_train)
    # Classify
    y = np_utils.to_categorical(y_test, 2)
    classesPredictionList = clf.predict(x_test) # 0 or 1
    classesProbaPredictionList = clf.predict_proba(x_test) # probability
    sensitivity, specificity, accuracy, precision, recall, F1_score, auc = metrics.generate_metrics(classesPredictionList,classesProbaPredictionList,y,verbose=False)
    
    if(save):
        joblib.dump(clf, "output/"+baseName+".pkl") 
        
    print("Accuracy: {:.4f}".format(accuracy))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))
    print("F1 Score: {:.4f}".format(F1_score))
    print("AUC: {:.4f}".format(auc))

def create_modelCV(x_train, y_train, x_test, y_test, numFolds= 5, c=1, k='linear'):
    """
    Model providing function:

    Create Keras model with SVM as classifier, compile test and generate metrics.
    """
    ### Cross-validation
    skf = StratifiedKFold(n_splits=numFolds, shuffle=True, random_state=1337)
    X = x_train
    Y = y_train
    sensitivitys, specificitys, accuracys, precisions, recalls, F1_scores, aucs = [[],[],[],[],[],[],[]]
    #kpbar = tqdm(total=numFolds, desc="Kfold", leave=False)
    y = np_utils.to_categorical(Y, 2)
    Y = numpy.array(Y)
    for train_index, test_index in skf.split(X, Y):
        ################# define SVM #################
        clf = svm.SVC(kernel = k, C = c, probability=True, random_state=1337)
        clf.fit(X[train_index], Y[train_index])
        # Classify
        classesPredictionList = clf.predict(X[test_index]) # 0 or 1
        classesProbaPredictionList = clf.predict_proba(X[test_index]) # probability
        sensitivity, specificity, accuracy, precision, recall, F1_score, auc = metrics.generate_metrics(classesPredictionList,classesProbaPredictionList,y[test_index],verbose=False)
        sensitivitys.append(sensitivity)
        specificitys.append(specificity)
        accuracys.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        F1_scores.append(F1_score)
        aucs.append(auc)
    
    sensitivitys = numpy.array(sensitivitys)
    specificitys = numpy.array(specificitys)
    accuracys = numpy.array(accuracys)
    precisions = numpy.array(precisions)
    recalls = numpy.array(recalls)
    F1_scores = numpy.array(F1_scores)
    aucs = numpy.array(aucs)
    print("Mean Accuracy: {:.4f} (+/- {:.4f})".format(accuracys.mean(), accuracys.std()))
    print("Mean Sensitivity: {:.4f} (+/- {:.4f})".format(sensitivitys.mean(), sensitivitys.std()))
    print("Mean Specificity: {:.4f} (+/- {:.4f})".format(specificitys.mean(), specificitys.std()))
    print("Mean F1 Score: {:.4f} (+/- {:.4f})".format(F1_scores.mean(), F1_scores.std()))
    print("Mean AUC: {:.4f} (+/- {:.4f})".format(aucs.mean(), aucs.std()))

if __name__ == '__main__':
    start_time = time.time()
    print("Loading dataset...")
    X_train, Y_train, X_test, Y_test = dataCV()
    x_train, y_train, x_test, y_test = data(X_train, Y_train)
    print("Training...")
    create_model(x_train, y_train, x_test, y_test, numFolds=5, c=10, k='rbf')
    print("Training with cross validation...")
    create_modelCV(X_train, Y_train, X_test, Y_test, numFolds=5, c=10, k='rbf')
    print("--- {:.1f} seconds ---".format(time.time() - start_time))
