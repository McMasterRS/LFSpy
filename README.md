# Localized Feature Selection (LFS)

- TODO one paragraph specifics on what this is about

## Getting Started

### Prerequisites
- TODO how to set up a sklearn pipeline
- TODO how to run the sklearn pipeline

### Installing
- TODO install dependencies (sklearn, python environment)

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

## Built With
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)

## Contributing

## Versioning

## Authors

## License

## Acknowledgments

1. N. Armanfard, JP. Reilly, and M. Komeili, "Local Feature Selection for Data Classification", IEEE Trans. on Pattern Analysis and Machine Intelligence, vol. 38, no. 6, pp. 1217-1227, 2016.
2. N. Armanfard, JP. Reilly, and M. Komeili, "Logistic Localized Modeling of the Sample Space for Feature Selection and Classification", IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 5, pp. 1396-1413, 2018.