# Localized Feature Selection (LFS)

Full documentation can be found at: [lfspy.readthedocs.io](lfspy.readthedocs.io)

Localized feature selection (LFS) is a supervised machine learning approach for embedding localized feature selection in classification. The sample space is partitioned into overlapping regions, and subsets of features are selected that are optimal for classification within each local region. As the size and membership of the feature subsets can vary across regions, LFS is able to adapt to local variation across the entire sample space.

This repository contains a python implementation of this method that is compatible with scikit-learn pipelines. For a Matlab version, refer to [https://github.com/armanfn/LFS](https://github.com/armanfn/LFS)

### Statement of Need

LFSpy offers an implementation of the Local Feature Selection (LFS) algorithm that is compatible with scikit-learn, one of the most widely used machine learning packages today. LFS combines classification with feature selection, and distinguishes itself by it flexibility in selecting a different subset of features for different data points based on what is most discriminative in local regions of the feature space. This means LFS overcomes a well-known weakness of many classification algorithms, i.e., classification for non-stationary data where the number of features is high relative to the number of samples. 

## Installation

```bash
pip install lfspy
```

### Dependancies
LFS requires:
* Python 3
* [NumPy](https://numpy.org/)>=1.14
* [SciPy](https://www.scipy.org/)>=1.1
* [Scikit-learn](https://scikit-learn.org/stable/index.html)>=0.18.2
* [pytest](https://docs.pytest.org/en/latest/)>=5.0.0

### Testing
We recommend running the provided test after installing LFSpy to ensure the results obtained match expected outputs.

`pytest` may be installed either directly through pip (`pip install pytest`) or using the `test`
extra (`pip install LFSpy[test]`).

```bash
pytest --pyargs LFSpy
```

This will output to console whether or not the results of LFSpy on two datasets (the sample dataset provided in this repository, and scikit-learn's Fisher Iris dataset) are exactly as expected.

So far, LFSpy has been tested on Windows 10 with and without Conda, and on Ubuntu. In all cases, results have been exactly the expected results.

## Usage
To use LFSpy on its own:
```python
from LFSpy import LocalFeatureSelection

lfs = LocalFeatureSelection()
lfs.fit(training_data, training_labels)
predicted_labels = lfs.predict(testing_data)
total_error, class_error = lfs.score(testing_data, testing_labels)
```

To use LFSpy as part of an sklearn pipeline:
```python
from LFS import LocalFeatureSelection
from sklearn.pipeline import Pipeline

lfs = LocalFeatureSelection()
pipeline = Pipeline([('lfs', lfs)])
pipeline.fit(training_data, training_labels)
predicted_labels = pipeline.predict(testing_data)
total_error, class_error = pipeline.score(testing_data, testing_labels)
```

### Tunable Parameters
* `alpha`: (default: 19) the maximum number of selected features for each representative point
* `gamma`: (default: 0.2) impurity level tolerance, controls proportion of out-of-class samples can be in local region
* `tau`: (default: 2) number of passes through the training set
* `sigma`: (default: 1) adjusts weightings for observations based on their distance, values greater than 1 result in lower weighting
* `n_beta`: (default: 20) number of beta values to test, controls the relative weighting of intra-class vs. inter-class distance in the objective function
* `nrrp`: (default: 2000) number of iterations for randomized rounding process
* `knn`: (default: 1) number of nearest neighbours to compare for classification

### Example
This example uses the sample data ([matlab_Data.mat](https://github.com/McMasterRS/LFSpy/blob/master/LFSpy/tests/matlab_Data.mat)) available in the [LFSpy/tests](https://github.com/McMasterRS/LFSpy/tree/master/LFSpy/tests) folder. The full example can be found in [example.py](https://github.com/McMasterRS/LFSpy/blob/master/example.py). On our test system, the fnial output prints the statement, "LFS test accuracy: 0.7962962962962963".

The code provided in [comparisons.py]{https://github.com/McMasterRS/LFSpy/blob/master/LFSpy/comparisons/comparisons.py) serve as additional examples of how to use LFSpy.

```python
import numpy as np
from scipy.io import loadmat
from LFSpy import LocalFeatureSelection
from sklearn.pipeline import Pipeline

mat = loadmat('LFSpy/tests/matlab_Data')
x_train = mat['Train'].T
y_train = mat['TrainLables'][0]
x_test = mat['Test'].T
y_test = mat['TestLables'][0]
        
print('Training and testing an LFS model with default parameters.\nThis may take a few minutes...')
lfs = LocalFeatureSelection(rr_seed=777)
pipeline = Pipeline([('classifier', lfs)])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
score = pipeline.score(x_test, y_test)
print('LFS test accuracy: {}'.format(score))
```

## Contribution Guidelines

Please see our [Contribution Guidelines](https://lfspy.readthedocs.io/en/latest/Community%20Guidelines.html) page.


## Authors
*  Oliver Cook
*  Kiret Dhindsa
*  Areeb Khawajaby
*  Ron Harwood
*  Thomas Mudway

## Acknowledgments

1. N. Armanfard, JP. Reilly, and M. Komeili, "Local Feature Selection for Data Classification", IEEE Trans. on Pattern Analysis and Machine Intelligence, vol. 38, no. 6, pp. 1217-1227, 2016.
2. N. Armanfard, JP. Reilly, and M. Komeili, "Logistic Localized Modeling of the Sample Space for Feature Selection and Classification", IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 5, pp. 1396-1413, 2018.
