# CNN for Fault Recognition
This repository include code and some supplemental files for paper: Deep Convolutional Neural Network for Automatic Fault Recognition from 3D Seismic Dataset

# Code 
Run train.ipynb to train the DCNN models. 

Model_zoo contain four different DCNN models included in this paper.

functions.py include some extra functions.

pytorchtools.py is used for early stopping.

Best checkpoints for each model are stored in the checkpoints folder.

savePredNpy_thebetest.ipynb is used to merge and save model predictions.

py-bsds500 is a modified version of repository: https://github.com/Britefury/py-bsds500
This folder is a python version evaluation method of the standard BSDS 500 edge detection dataset. 

requirement folder list all required packages

augmentation_examples.ipynb can used to generate different augmentation examples, they would help you understand the impact of data augmentation 

# Comparative Results
Comparative results with two related works (Wu et al's faultSeg3D model and Cunha et al's Transfer learning model) are also made aviable to illustrated how we compare our work with their works. 

Comprison with Wu et al's faultSeg3D model is stored in faultSeg folder
    
    we modified prediction.ipynb, predNew.ipynb, train.py
    
    we added prepare_3Dcube_Thebe_Dataset.ipynb and trianThebe.out 

comprison with Cunha et al's transfer learning model is store in SFD-CNN-TL folder
    
    we added folder/file: finetune.ipynb, predictNew.ipynb, classifyAndMetricsGSB-compare.ipynb, GSB_predictions, gsbData, xl2800realgt.npy

# Dataset
The dataset used in this paper is a multi-megabytes dataset, please download it through the link provided in the paper. To access the original dataset, please check our data paper "A gigabyte interpreted seismic dataset for automatic fault recognition"

more about converting segy to numpy can be found in link: https://github.com/anyuzoey/SEGY2NUMPY 


# License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details



