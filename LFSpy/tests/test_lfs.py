import numpy as np
from scipy.io import loadmat
from LFSpy import LocalFeatureSelection
from sklearn.pipeline import Pipeline
from sklearn import datasets

def load_dataset(name):
    '''
    Loads a test/demo dataset.
    '''
    print('Loading dataset ' + name + '...')
    if name is 'sample':
        mat = loadmat('matlab_Data')
        training_data = mat['Train']
        training_labels = mat['TrainLables'][0]
        testing_data = mat['Test']
        testing_labels = mat['TestLables'][0]
        
    elif name is 'iris':
        # we only take the first two classes for binary classification
        train_idx = np.arange(0, 100, 2)
        test_idx = np.arange(1, 100, 2)
        
        iris = datasets.load_iris()
        training_data = iris.data[train_idx,:]
        training_labels = iris.target[train_idx]
        testing_data = iris.data[test_idx,:]
        testing_labels = iris.target[test_idx]
    
    return training_data, training_labels, testing_data, testing_labels
    
def train_model(x_train, y_train, x_test, y_test):
    '''
    Trains an tests and LFS model using default parameters on the given dataset.
    '''
    print('Training and testing an LFS model with default parameters.\nThis may take a few minutes...')
    lfs = LocalFeatureSelection() 
    pipeline = Pipeline([('classifier', lfs)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)
    
    return score[0][0][0], y_pred

def test_sample_data():

    training_data, training_labels, testing_data, testing_labels = load_dataset('sample')
    score, y_pred = train_model(training_data, training_labels, testing_data, testing_labels)

    expected_score = 20.370370370370374
    expected_preds = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                        0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    score_diff = np.abs(score - expected_score)
    pred_diff = np.mean(np.abs(y_pred - expected_preds))

    assert score == expected_score
    assert np.array_equal(y_pred, expected_preds)

def test_iris_data():

    training_data, training_labels, testing_data, testing_labels = load_dataset('iris')
    score, y_pred = train_model(training_data.T, training_labels, testing_data.T, testing_labels)

    expected_score = 0.
    expected_preds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1]

    assert score == expected_score
    assert np.array_equal(y_pred, expected_preds)