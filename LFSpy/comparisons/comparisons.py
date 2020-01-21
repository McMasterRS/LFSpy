import numpy as np
import warnings
from scipy.io import loadmat
from LFSpy import LocalFeatureSelection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn import datasets
import matplotlib.pyplot as plt


def load_dataset(name):
    '''
    Loads a test/demo dataset.
    '''
    print('Loading dataset ' + name + '...')
    if name is 'sample':
        mat = loadmat('LFSpy/tests/matlab_Data')
        training_data = mat['Train'].T
        training_labels = mat['TrainLables'][0]
        testing_data = mat['Test'].T
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
    
def results_lfspy(x_train, y_train, x_test, y_test):
    '''
    Trains an tests and LFS model using default parameters on the given dataset.
    '''
    print('Training and testing an LFS model with default parameters.\nThis may take a few minutes...')
    lfs = LocalFeatureSelection(rr_state=777)
    pipeline = Pipeline([('classifier', lfs)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)
    
    return score, y_pred

def results_rforest(x_train, y_train, x_test, y_test):
    print('Training and testing a Random Forest with default parameters.')
    rfc = RandomForestClassifier(random_state=777)
    pipeline = Pipeline([('classifier', rfc)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)
    
    return score, y_pred

def results_fsvm(x_train, y_train, x_test, y_test):
    print('Training and testing a SVM with default parameters with F-stat feature selection (25% of features selected).')
    svm = LinearSVC(random_state=777)
    sel = SelectKBest(f_classif, k=int(0.25*x_train.shape[1]))
    pipeline = Pipeline([('feat_sel', sel), ('classifier', svm)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    score = pipeline.score(x_test, y_test)
    
    return score, y_pred

training_data, training_labels, testing_data, testing_labels = load_dataset('sample')
score_lfs, y_pred_lfs = results_lfspy(training_data, training_labels, testing_data, testing_labels)
score_rfc, y_pred_rfc = results_rforest(training_data, training_labels, testing_data, testing_labels)
score_svm, y_pred_svm = results_fsvm(training_data, training_labels, testing_data, testing_labels)

training_data, training_labels, testing_data, testing_labels = load_dataset('iris')
score_lfs_iris, y_pred_lfs_iris = results_lfspy(training_data, training_labels, testing_data, testing_labels)
score_rfc_iris, y_pred_rfc_iris = results_rforest(training_data, training_labels, testing_data, testing_labels)
score_svm_iris, y_pred_svm_iris = results_fsvm(training_data, training_labels, testing_data, testing_labels)

# Plot the comparison of results
scores = [score_lfs, score_rfc, score_svm]
scores_iris = [score_lfs_iris, score_rfc_iris, score_svm_iris]

def plotScores(scores, title=None):
    '''
    Plot classification scores.
    '''
    plt.figure()
    plt.bar(['LFS','RFC','SVM'], scores)
    plt.ylim([0,1])    

    for i, v in enumerate(scores):
        plt.text(i - 0.1, 0.4, '{:.{}f}'.format(v,2), size=12)
    
    plt.title(title, fontsize=14)
    plt.savefig(title+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    return None

plotScores(scores, 'Sample Data Classification Accuracies')
plotScores(scores_iris, 'Iris Data Classification Accuracies')