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

from keras import backend as K

K.set_image_dim_ordering('tf')

import pandas as pd
import cv2, numpy
numpy.random.seed(1337)

from keras.models import model_from_json

def main():
    
    import time
    
    #  params
    section_dir = "classification/"
    section_name = "inl256_region_smoothed_clipped.csv"
    patch_size = 20
    pixel_step = 3
    half_patch = int(patch_size/2)
    resize = 45 
    imageChannels = 1
    
    # create output directory
    directory = "classification/output/"
        
    # load CNN
    jsonModelFilePath = "base_model/model.json"
    weightsFilePath =  "base_model/model.h5"
    jsonModelFile = open(jsonModelFilePath, 'r' )
    jsonModel = jsonModelFile.read()
    jsonModelFile.close()
    model = model_from_json(jsonModel)
    model.load_weights(weightsFilePath)
    #------ delete last layers of base model -------
    for i in range (7): # todo : read from config
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
    
    #load svm
    svmModelPath = "output/model_TL_FESVM.pkl"
    from sklearn.externals import joblib
    clf = joblib.load(svmModelPath) 
   
    # read section
    df = pd.read_csv(section_dir + section_name, delimiter=' ', header = None)
    df_mat = df.values
    
    print("Classifying patches...")
    
    # get section info
    nb_rows = df.shape[0] 
    nb_cols = df.shape[1] 
    
    count_patches = 0
    patch_name_list = []
    prediction_list = []
    start_time = time.time()
    for i in range (half_patch, nb_rows - half_patch, pixel_step):
        for j in range (half_patch, nb_cols - half_patch, pixel_step):
            # create patch
            start_row = i - half_patch
            start_col = j - half_patch
            patch_list = []
            patch =  numpy.zeros((patch_size,patch_size))
            for x in range(patch_size):
                for y in range(patch_size):
                    patch[x][y] = df_mat[start_row + x][start_col + y]
            # resize, clip
            patch = cv2.resize(patch, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
            patch = numpy.clip(patch, -1., 1.)
            patch_list.append(patch)
            # format
            patch = numpy.array( patch_list ) 
            patch = patch.reshape( patch.shape[0], resize, resize, imageChannels )
            patch = patch.astype( 'float32' )
            # classify
            print("patch " + str(count_patches))
            classesPredictionList = []
            features = model.predict(patch)
            classesPredictionList = clf.predict(features)
            
            # write
            patch_name = 'patch_p_' + str(i) + '_' + str(j) + '.csv'
            patch_name_list.append(patch_name)
            prediction_list.append(classesPredictionList[0])
            # count
            count_patches +=1

    print("--- %s seconds ---" % (time.time() - start_time))

    ######################################################################################

    # save
    print("Writing classification file...")
    predictionsFile = open(directory + 'classification_' + section_name.split('.')[0] + '.txt', 'w')
    for i in range(0, len(prediction_list)):
        patch_name = patch_name_list[i]
        prediction = prediction_list[i]
        predictionsFile.write( patch_name + " " + str(prediction) + "\n" )
    predictionsFile.close()
        
    
#####################################
main()