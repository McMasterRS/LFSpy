.. LFSpy documentation master file, created by
   sphinx-quickstart on Thu Jan 23 11:23:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LFSpy's documentation!
=================================

Localized feature selection (LFS) is a supervised machine learning approach for embedding localized feature selection in classification. The sample space is partitioned into overlapping regions, and subsets of features are selected that are optimal for classification within each local region. As the size and membership of the feature subsets can vary across regions, LFS is able to adapt to local variation across the entire sample space.

This repository contains a python implementation of this method that is compatible with scikit-learn pipelines. For a Matlab version, refer to https://github.com/armanfn/LFS

Installation
=================================
pip install lfspy

Dependancies
=================================
LFS requires:

Python 3
NumPy>=1.14
SciPy>=1.1
Scikit-learn>=0.18.2
pytest>=5.0.0

Testing
=================================
We recommend running the provided test after installing LFSpy to ensure the results obtained match expected outputs.

pytest may be installed either directly through pip (pip install pytest) or using the test extra (pip install LFSpy[test]).

pytest --pyargs LFSpy
This will output to console whether or not the results of LFSpy on two datasets (the sample dataset provided in this repository, and scikit-learn's Fisher Iris dataset) are exactly as expected.

So far, LFSpy has been tested on Windows 10 with and without Conda, and on Ubuntu. In all cases, results have been exactly the expected results.

Usage
=================================
To use LFSpy on its own:

from LFSpy import LocalFeatureSelection

lfs = LocalFeatureSelection()
lfs.fit(training_data, training_labels)
predicted_labels = lfs.predict(testing_data)
total_error, class_error = lfs.score(testing_data, testing_labels)
To use LFSpy as part of an sklearn pipeline:

from LFS import LocalFeatureSelection
from sklearn.pipeline import Pipeline

lfs = LocalFeatureSelection()
pipeline = Pipeline([('lfs', lfs)])
pipeline.fit(training_data, training_labels)
predicted_labels = pipeline.predict(testing_data)
total_error, class_error = pipeline.score(testing_data, testing_labels)
Tunable Parameters
alpha: (default: 19) the maximum number of selected features for each representative point
gamma: (default: 0.2) impurity level tolerance, controls proportion of out-of-class samples can be in local region
tau: (default: 2) number of passes through the training set
sigma: (default: 1) adjusts weightings for observations based on their distance, values greater than 1 result in lower weighting
n_beta: (default: 20) number of beta values to test, controls the relative weighting of intra-class vs. inter-class distance in the objective function
nrrp: (default: 2000) number of iterations for randomized rounding process
knn: (default: 1) number of nearest neighbours to compare for classification

Authors
=================================
Oliver Cook
Kiret Dhindsa
Areeb Khawajaby
Ron Harwood
Thomas Mudway

Acknowledgments
=================================
N. Armanfard, JP. Reilly, and M. Komeili, "Local Feature Selection for Data Classification", IEEE Trans. on Pattern Analysis and Machine Intelligence, vol. 38, no. 6, pp. 1217-1227, 2016.
N. Armanfard, JP. Reilly, and M. Komeili, "Logistic Localized Modeling of the Sample Space for Feature Selection and Classification", IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 5, pp. 1396-1413, 2018.


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
