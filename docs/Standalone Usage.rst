Usage
=====
To use LFSpy on its own::

    from LFSpy import LocalFeatureSelection

    lfs = LocalFeatureSelection()
    lfs.fit(training_data, training_labels)
    predicted_labels = lfs.predict(testing_data)
    total_error, class_error = lfs.score(testing_data, testing_labels)
