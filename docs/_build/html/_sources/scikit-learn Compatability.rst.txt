scikit-learn Compatability
=================================

To use LFSpy as part of an sklearn pipeline::

    from LFS import LocalFeatureSelection
    from sklearn.pipeline import Pipeline

    lfs = LocalFeatureSelection()
    pipeline = Pipeline([('lfs', lfs)])
    pipeline.fit(training_data, training_labels)
    predicted_labels = pipeline.predict(testing_data)
    total_error, class_error = pipeline.score(testing_data, testing_labels)