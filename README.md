# Localized Feature Selection (LFS)

Localized feature selection (LFS) is an approach whereby each region of the sample space is associated with its own distinct optimized feature set, which may vary both in membership and size across the sample space. This allows the feature set to optimally adapt to local variations in the sample space.

This repository contains a python implementation of this method that is compatible with scikit-learn pipelines. For a Matlab version, refer to [https://github.com/armanfn/LFS](https://github.com/armanfn/LFS)

## Installation

```bash
pip install lfspy
```

### Dependancies
LFS requires:
*  python
* [NumPy](https://numpy.org/)
* [SciPy](https://www.scipy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/index.html)

## Usage

```python
from LFS import LocalFeatureSelection
from sklearn.pipeline import Pipeline

lfs = LocalFeatureSelection()
pipeline = Pipeline([('lfs', lfs)])
pipeline.fit(training_data, training_labels)
predicted_labels = pipeline.predict(testing_data)
total_error, class_error = pipeline.score(testing_data, testing_labels)
```

## Authors
*  Oliver Cook
*  Kiret Dhindsa
*  Areeb Khawajaby
*  Ron Harwood
*  Thomas Mudway

## Acknowledgments

1. N. Armanfard, JP. Reilly, and M. Komeili, "Local Feature Selection for Data Classification", IEEE Trans. on Pattern Analysis and Machine Intelligence, vol. 38, no. 6, pp. 1217-1227, 2016.
2. N. Armanfard, JP. Reilly, and M. Komeili, "Logistic Localized Modeling of the Sample Space for Feature Selection and Classification", IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 5, pp. 1396-1413, 2018.
