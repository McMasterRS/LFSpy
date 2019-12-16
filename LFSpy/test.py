import numpy as np
import warnings
from scipy.io import loadmat
from .LFS import LocalFeatureSelection
from sklearn.pipeline import Pipeline
from sklearn import datasets
import pkg_resources
from pathlib import Path

def load_dataset(name):
    '''
    Loads a test/demo dataset.
    '''
    print('Loading dataset ' + name + '...')
    if name is 'sample':
        mat = loadmat(Path(__file__).parent / 'matlab_Data.mat')
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
    lfs = LocalFeatureSelection(rr_seed=20) 
    pipeline = Pipeline([('classifier', lfs)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)
    
    return score[0][0][0], y_pred

def verify_output(model_out, dataset_name):
    '''
    Compares the outputs of a model to the expected output stored from a run
    performed on the development hardware using known outcomes. The expected
    output is considered "correct", but due to hardware and environment 
    differences, we allow for small variation and only issue a warning if the
    output is not exact. The expected score is therefore hardcoded here.
    '''
    score, y_pred = model_out
    if dataset_name is 'sample':
        print('-------------DATASET = SAMPLE-------------')
        expected_score = 20.370370370370374
        expected_preds = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                          0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                          0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
    elif dataset_name is 'iris':
        print('-------------DATASET = IRIS-------------')
        expected_score = 0.
        expected_preds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1]
        
    score_diff = np.abs(score - expected_score)
    pred_diff = np.mean(np.abs(y_pred - expected_preds))
    
    if score == expected_score:
        print('Computed score is exactly the expected score.')
    elif score_diff < 0.02 * expected_score:
        warnings.warn('Computed score is not exactly the expected score, but it is within 2 percent tolerance bound. This may be due to differences in hardware or software environment, but you may want to check your installation of LFSpy and look into the results to ensure that this small difference is acceptable.')
    else:
        warnings.warn('Computed score is not within 2 percent of the expected score. Check that the installation completed without error, and that test.py was not modified in any way. If needed, please refer to the documentation or contact the developers.')

    if np.array_equal(y_pred, expected_preds):
        print('Computed predictions are exactly the expected predictions.')
    elif pred_diff < 0.02:
        warnings.warn('Computed predictions are not exactly the expected predictions, but they are within 2 percent tolerance bound. This may be due to differences in hardware or software environment, but you may want to check your installation of LFSpy and look into the results to ensure that this small difference is acceptable.')
    else:
        warnings.warn('Computed predictions are not within 2 percent of the expected predictions. Check that the installation completed without error, and that test.py was not modified in any way. If needed, please refer to the documentation or contact the developers.')
    print('\n\n')

    return None

training_data, training_labels, testing_data, testing_labels = load_dataset('sample')
score, y_pred = train_model(training_data, training_labels, testing_data, testing_labels)
verify_output((score, y_pred), dataset_name='sample')

training_data, training_labels, testing_data, testing_labels = load_dataset('iris')
score, y_pred = train_model(training_data.T, training_labels, testing_data.T, testing_labels)
verify_output((score, y_pred), dataset_name='iris')