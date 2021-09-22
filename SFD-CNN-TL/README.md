*This page is available as an executable or viewable **Jupyter Notebook**:* 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/augustoicaro/SFD-CNN-TL/master?filepath=classifyAndViewGSB.ipynb)
[<img src="https://github.com/jupyter/design/blob/master/logos/Badges/nbviewer_badge.png?raw=true" alt="NbViwer" width="109"/>](https://nbviewer.jupyter.org/github/augustoicaro/SFD-CNN-TL/blob/master/classifyAndViewGSB.ipynb)

# This code allows to reproduce results of our article:
## Seismic fault detection in real data using Transfer Learning from a Convolutional Neural Network pre-trained with synthetic seismic data [https://doi.org/10.1016/j.cageo.2019.104344](https://doi.org/10.1016/j.cageo.2019.104344)

## Publications
If you find this work helpful in your research, please cite:
```
@article{cunha2020SFD,
title = "Seismic fault detection in real data using transfer learning from a convolutional neural network pre-trained with synthetic seismic data",
journal = "Computers & Geosciences",
volume = "135",
pages = "104344",
year = "2020",
issn = "0098-3004",
doi = "https://doi.org/10.1016/j.cageo.2019.104344",
url = "http://www.sciencedirect.com/science/article/pii/S0098300418307040",
author = "Augusto Cunha and Axelle Pochet and Hélio Lopes and Marcelo Gattass",
keywords = "Transfer learning, Convolutional neural network, Seismic fault"
}
```

### Notebook visualization
We provide two option to easily open and see our ipython notebooks:
- Visualize with [NbViwer](https://nbviewer.jupyter.org/github/augustoicaro/SFD-CNN-TL/blob/master/classifyAndViewGSB.ipynb)
- Visualize and modify with [Binder](https://mybinder.org/v2/gh/augustoicaro/SFD-CNN-TL/master?filepath=classifyAndViewGSB.ipynb)

We strongly recommend using NbViwer to visualize our notebooks instead open in GitHub, because you will see the interactive plots.

### Dependencies
	environment.yml
	
To install all requirements in the environment use:

	conda env create -f environment.yml
	
Sometimes is needed to create a kernel of the environment to jupyter notebooks:

	python -m ipykernel install --user --name sfd --display-name "SFD-CNN-TL"

### Our F3 Block pre-trained model can be found in:
 	base_model/model.json
 	base_model/model.h5
 	
### Our F3 Block slice manualy interpreted dataset can be found in:
 	dataset/fault
 	dataset/nonfault
 
### Transfer learning methods:
 	ft.py : full fine tuning (FFT)
 	mlp.py : feature extractor with Multi Layer Perceptron (FE-MLP)
 	svm.py : feature extractor with Support Vector Machine (FE-SVM)
 
Default parameters are set to produce the results presented in the article.
 
Generated models can be saved by setting the boolean value save=true in functions create_model(). they will be save in the output/ directory.
 	
### Classification results:
 	classify.py : generates a classification file for models saved as .json and .h5
 	classify_with_SVM.py : generates a classification file for models saved as .pkl
 	
Classification files is saved in directiry classification/output/
It contains patches coordinates associated to a class value (1 for fault, 0 otherwise)
 	
We provide a region of a real section where a fault is clearly visible as demo in the classification/ directory.
Other sections can be classified modifying the classify.py and classify_with_SVM.py files.
 	
### Metrics:
 	metrics.py : computes quality metrics (accuracy, sensitivity, specificity = recall, F1-score, ROC AUC and we added precision)
    
### Interactive examples:
	classifyAndViewF3.ipynb: Classify selected sections of F3 seismic data with all pretrained methods and show the results
	classifyAndViewGSB.ipynb: Classify selected sections of GSB seismic data with all pretrained methods and show the results
	TrainAndSave.ipynb: Train all methods with one interpreted section of a real data and save network weights
    


