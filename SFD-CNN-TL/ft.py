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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import metrics

def save_model(model, filename):
    model_json = model.to_json() 
    with open("output/" + filename + ".json", "w") as json_file: json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("output/" + filename + "_weights.h5") 

def load_model(modelJsonPath = 'base_model/model.json', modelWeightsPath = 'base_model/model.h5') :
    ################# load base model #################
    jsonFile = open(modelJsonPath, 'r')
    loadedModelJson = jsonFile.read()
    jsonFile.close()
    base_model = model_from_json(loadedModelJson)
    base_model.load_weights(modelWeightsPath)

    return base_model

def data(X_train, Y_train, numberOfClasses = 2):
    Y_train = np_utils.to_categorical(Y_train, numberOfClasses)
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, random_state=1337)
    return x_train, y_train, x_test, y_test    

def dataCV(trainFaultDirectory='dataset/fault/',trainNonFaultDirectory='dataset/nonfault/'):
    
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
    x_train = X
    y_train = Y
    x_test = []
    y_test = []
    
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test, numberOfClasses=2, numberOfEpochs = 10, batchSize = 30, save=True, baseName='fftModel'):
    """
    Model providing function:

    Create Keras model with MLP as classifier, compile test and generate metrics.
    """
    base_model = load_model()

    # Compile
    base_model.compile( loss='binary_crossentropy', optimizer= 'sgd', metrics=['accuracy'] )

    # Train
    base_model.fit(x_train,
                   y_train,
                   batch_size = batchSize,
                   epochs = numberOfEpochs,
                   verbose = 0,
                   validation_data=(x_test, y_test))  
  
    # Classify
    classesPredictionList = base_model.predict_classes(x_test, verbose=0) # 0 or 1
    classesProbaPredictionList = base_model.predict_proba(x_test) # probability
    sensitivity, specificity, accuracy, precision, recall, F1_score, auc = metrics.generate_metrics(classesPredictionList,classesProbaPredictionList,y_test,verbose=False)

    # Save Model
    if(save):
        save_model(base_model, baseName)
    
    print("Accuracy: {:.4f}".format(accuracy))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))
    print("F1 Score: {:.4f}".format(F1_score))
    print("AUC: {:.4f}".format(auc))

def create_modelCV(x_train, y_train, x_test, y_test, numFolds= 5, numberOfClasses=2, numberOfEpochs = 10, batchSize = 30):
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

    for train_index, test_index in skf.split(X, Y):
        
        base_model = load_model()

        # Compile
        base_model.compile( loss='binary_crossentropy', optimizer= 'sgd', metrics=['accuracy'] )
        
        # Train
        base_model.fit(X[train_index],
                       y[train_index],
                       batch_size = batchSize,
                       epochs = numberOfEpochs,
                       verbose = 0,
                       validation_data=(X[test_index], y[test_index]))
        # Classify
        classesPredictionList = base_model.predict_classes(X[test_index], verbose=0) # 0 or 1
        classesProbaPredictionList = base_model.predict_proba(X[test_index]) # probability
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
    create_model(x_train, y_train, x_test, y_test, numberOfEpochs = 20, save=True, baseName='fftModel')
    print("Training with cross validation...")
    create_modelCV(X_train, Y_train, X_test, Y_test, numFolds=5, numberOfEpochs = 20)
    print("--- {:.1f} seconds ---".format(time.time() - start_time))
