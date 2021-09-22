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

from tensorflow.keras import backend as K
# K.set_image_dim_ordering('tf')
# K.set_image_data_format('channels_last')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
K.set_session(tf.Session(config=config))

import gc
import cv2, os, numpy, sys
import pandas as pd
import multiprocessing
import time

from tensorflow.keras.models import model_from_json, Sequential
from joblib import Parallel, delayed


numpy.random.seed(1337)

# Reset Keras Session
def reset_keras(base_model, model):
    sess = K.get_session()
    K.clear_session()
    sess.close()
    sess = K.get_session()

    try:
        del base_model
        del model
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #config.gpu_options.visible_device_list = "0"
    K.set_session(tf.Session(config=config))

def processPatches(data, patch_size, pixel_step, resize, nb_channels):
    # get data
    if isinstance(data, pd.DataFrame):
        section_mat = data.values
    else:
        section_mat = data
    half_patch = int(patch_size/2)
    
    # get image info
    nb_rows = data.shape[0] 
    nb_cols = data.shape[1]
    #print(nb_rows)
    #print(nb_cols)
    
    count_patches = 0
    patch_name_list = []
    patch_list = []
    for i in range (half_patch, nb_rows - half_patch, pixel_step):
        for j in range (half_patch, nb_cols - half_patch, pixel_step):
            # create patch
            start_row = i - half_patch
            start_col = j - half_patch
            patch =  numpy.zeros((patch_size,patch_size)) # 1 empty patch
            for x in range(patch_size):
                for y in range(patch_size):
                    patch[x][y] = section_mat[start_row + x][start_col + y]
            # resize, clip
            patch = cv2.resize(patch, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
            patch = numpy.clip(patch, -1., 1.)
            # append to global list
            patch_list.append(patch)
            patch_name = 'patch_p_' + str(i) + '_' + str(j) + '.csv'
            patch_name_list.append(patch_name)
            # count
            count_patches +=1
    
    return patch_list, patch_name_list

def classify(input_dir, patch_size, resize_size, pixel_step, jsonModelFilePath, weightsFilePath, modelName):
    start_time = time.time()
    
    # set params
    resize = resize_size 
    imageChannels = 1
    
    # create output directory
    directory = "output/classification/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # read all files in 1 step
    df_list = []
    df_names = []
    files = os.listdir(input_dir)
    for i in range(0, len(files)):
        filename = files[i]
        section_name = filename.split('_')[0]
        df = pd.read_csv(input_dir + filename, delimiter=' ', header = None)
        df_list.append(df)
        df_names.append(modelName + "_" + section_name)
        
    # load model
    jsonModelFile = open(jsonModelFilePath, 'r' )
    base_model = jsonModelFile.read()
    jsonModelFile.close()
    model = model_from_json(base_model)
    model.load_weights(weightsFilePath)
    model.compile( loss='binary_crossentropy', optimizer='sgd', metrics=[ 'accuracy' ] )
    
    # prepare save prediction for all sections
    nb_sections = len(df_list)
    
    # create patches in parrallel 
    s = 0
    nb_section = len(df_list)
    section_step = 4
    while 1 :
        s_init = s
        
        if(s == nb_section):
            break
        
        if(s+section_step > len(df_list)):
            df_sub_list = df_list[s:len(df_list)]
            s = nb_section
        else:
            df_sub_list = df_list[s:s+section_step]
            s = s+section_step
        
        print("Creating patches...")
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose = 100)(delayed(processPatches)(i,  patch_size, pixel_step, resize, imageChannels) for i in df_sub_list)
        current_sections_patch_lists, current_sections_patch_name_lists = zip(*results)
    
        # classify and save
        current_sections_prediction_lists = []
        for i in range(0, len(current_sections_patch_lists)):
            print("Classifying section " + str(s_init + i + 1) + "/" + str(nb_section))
            patch_list = current_sections_patch_lists[i]
            patches = numpy.array( patch_list ) 
            patches = patches.reshape( patches.shape[0], resize, resize, imageChannels)
            patches = patches.astype( 'float32' )
            # classify
            classesPredictionList = []
            classesPredictionList = model.predict_classes(patches)
            current_sections_prediction_lists.append(classesPredictionList)

        print("Writing classification files...")
        for i in range(0, len(current_sections_patch_lists)):
            print("Section " + df_names[s_init + i])
            predictionsFile = open(directory + 'classification_' + df_names[s_init + i] + '.txt', 'w')
            for j in range(0, len(current_sections_prediction_lists[i])):
                patch_name = current_sections_patch_name_lists[i][j]
                prediction = current_sections_prediction_lists[i][j]
                predictionsFile.write( patch_name + " " + str(prediction) + "\n" )
            predictionsFile.close()
               
    reset_keras(base_model,model)
    print("--- %s seconds ---" % (time.time() - start_time))
    
def classifySVM(input_dir, patch_size, resize_size, pixel_step, jsonModelFilePath, weightsFilePath, modelName, svmModelPath):
    start_time = time.time()

    # set params
    resize = resize_size # todo = read from json model
    imageChannels = 1
    
    # create output directory
    directory = "output/classification/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # read all files in 1 step
    df_list = []
    df_names = []
    files = os.listdir(input_dir)
    for i in range(0, len(files)):
        filename = files[i]
        section_name = filename.split('_')[0]
        df = pd.read_csv(input_dir + filename, delimiter=' ', header = None)
        df_list.append(df)
        df_names.append(modelName + "_" + section_name)
        
    # load CNN
    jsonModelFile = open(jsonModelFilePath, 'r' )
    jsonModel = jsonModelFile.read()
    jsonModelFile.close()
    base_model = model_from_json(jsonModel)
    base_model.load_weights(weightsFilePath)
    #------ delete last layers -------
    model = Sequential(base_model.layers[:-7])
    
    #------ Load SVM
    from sklearn.externals import joblib
    clf = joblib.load(svmModelPath) 
    
    # prepare save prediction for all sections
    nb_sections = len(df_list)
    
    # create patches in parrallel 
    s = 0
    nb_section = len(df_list)
    section_step = 4
    while 1 :
        
        s_init = s
        
        if(s == nb_section):
            break
        
        if(s+section_step > len(df_list)):
            df_sub_list = df_list[s:len(df_list)]
            s = nb_section
        else:
            df_sub_list = df_list[s:s+section_step]
            s = s+section_step
        
        print("Creating patches...")
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose = 100)(delayed(processPatches)(i,  patch_size, pixel_step, resize, imageChannels) for i in df_sub_list)
        current_sections_patch_lists, current_sections_patch_name_lists = zip(*results) # returns tuples :/
    
    
        # classify and save
        current_sections_prediction_lists = []
        for i in range(0, len(current_sections_patch_lists)):
            print("Classifying section " + str(s_init + i + 1) + "/" + str(nb_section))
            patch_list = current_sections_patch_lists[i]
            patches = numpy.array( patch_list ) 
            patches = patches.reshape( patches.shape[0], resize, resize, imageChannels)
            patches = patches.astype( 'float32' )
            # classify
            classesPredictionList = []
            features = model.predict(patches)
            classesPredictionList = clf.predict(features)
            current_sections_prediction_lists.append(classesPredictionList)

        print("Writing classification files...")
        for i in range(0, len(current_sections_patch_lists)):
            print("Section " + df_names[s_init + i])
            predictionsFile = open(directory + 'classification_' + df_names[s_init + i] + '.txt', 'w')
            for j in range(0, len(current_sections_prediction_lists[i])):
                patch_name = current_sections_patch_name_lists[i][j]
                prediction = current_sections_prediction_lists[i][j]
                predictionsFile.write( patch_name + " " + str(prediction) + "\n" )
            predictionsFile.close()
    
    reset_keras(base_model,model)
    print("--- %s seconds ---" % (time.time() - start_time))



