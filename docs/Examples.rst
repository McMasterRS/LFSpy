Examples
=================================
Given here is an example demonstration of localized feature selection and LFSpy for feature selection and classification using the common Iris flower data set.

For installation instructions please refer to the "Installation" section.


::

    import numpy as np
    from scipy.io import loadmat
    from LFSpy import LocalFeatureSelection
    from sklearn.pipeline import Pipeline

    # Loads the sample dataset
    mat = loadmat('LFSpy/tests/matlab_Data')
    x_train = mat['Train'].T
    y_train = mat['TrainLables'][0]
    x_test = mat['Test'].T
    y_test = mat['TestLables'][0]
            

    #Trains an tests and LFS model using default parameters on the given dataset.
    print('Training and testing an LFS model with default parameters.\nThis may take a few minutes...')
    lfs = LocalFeatureSelection(rr_seed=777)
    pipeline = Pipeline([('classifier', lfs)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)
    print('LFS test accuracy: {}'.format(score))
    # On our test system, running this code prints the following: LFS test accuracy: 0.7962962962962963