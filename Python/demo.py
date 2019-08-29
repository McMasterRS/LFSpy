from scipy.io import loadmat
from LFS import LocalFeatureSelection
from sklearn.pipeline import Pipeline

mat = loadmat('./matlab_Data')

training_data = mat['Train']
training_labels = mat['TrainLables']
testing_data = mat['Test']
testing_labels = mat['TestLables']

X = training_data
y = training_labels[0]
X_test = testing_data
y_test = testing_labels[0]
lfs = LocalFeatureSelection()
pipeline = Pipeline([('lfs', lfs)])

pipeline.fit(X, y)
Y = pipeline.predict(X_test)
pipeline.score(y_test)