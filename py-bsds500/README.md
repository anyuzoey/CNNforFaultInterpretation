# Python port of BSDS 500 boundary prediction evaluation suite

Uses quite a lot of code from the original BSDS evaluation suite at
(https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)[https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/]

Takes the original C++ source code that provides the `matchPixels` function for Matlab
and wraps it with Cython to make it available from Python.

Provides a Python implementation of the morphological thinning operation.

Compile the extension module with:

`python setup.py build_ext --inplace`

Then run:

`evalnpy.py` to produce evaluation results. In this file, you should fill in the location of annotation files and model_name, which will give the location of the prediction files.