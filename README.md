# CNN for Fault Interpretation
This repository include code and some supplemental files for paper: Building Realistic Structure Models to Train Convolutional Neural Networks for Seismic Structural Interpretation

# Code 
Run train900200.py or train-step2.py to train the DCNN models.
Model_zoo contain four different DCNN models included in this paper.
functions.py include some extra functions.
pytorchtools.py is used for early stopping.

model checkpoints are stored in the checkpoints folder.

readSGY.ipynb explains how we convert the SGY file to numpy array file.
savePredNpy_thebetest.ipynb is used to merge and save model predictions.

py-bsds500 is a modified version of repository: https://github.com/Britefury/py-bsds500
This folder is a python version evaluation method of the standard BSDS 500 edge detection dataset.


# License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details



