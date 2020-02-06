---
title: '``LFSpy``: A Python Implementation of Local Feature Selection for Data Classification with ``scikit-learn`` Compatibility'
tags: 
    - Python
    - Machine Learning
    - Feature Selection
    - Classification
    - Data Science
authors:
 -  name: Kiret Dhindsa
    orcid: 0000-0003-4849-732X
    affiliation: "1, 2, 3"
 -  name: Oliver Cook
    orcid: 0000-0002-5511-094X
    affiliation: "1"
 -  name: Thomas Mudway
    orcid: 0000-0001-6213-2111
    affiliation: "1"
 -  name: Areeb Khawaja
    orcid: 0000-0003-4528-9146
    affiliation: "1"
 -  name: Ron Harwood
    orcid: 0000-0002-7922-0641
    affiliation: "1"
 -  name: Ranil Sonnadara
    orcid: 0000-0001-8318-5714
    affiliation: "1, 2, 3"
affiliations:
 - name: Research and High Performance Computing, McMaster University
   index: 1
 - name: Vector Institute
   index: 2
 - name: Department of Surgery, McMaster University
   index: 3
date: 12 December 2019
bibliography: paper.bib
---

# Background
Successful machine learning depends on inputting features, or variables, into an algorithm that provide information that is useful for solving the problem at hand. For supervised classification problems in machine learning, where the goal is to label or categorize new data based on patterns identified in labeled training data, this means that the features used to train a machine learning model must help discriminate between different categories of data samples. However, it is not always possible to know a priori which of the available features are informative, and which are not. The presence of uninformative features can contribute noise and reduce the robustness and performance of classification models. Therefore, an important step in machine learning is the selection of informative features and the omission of uninformative features.

# Summary
Where typical feature selection methods find an optimal feature subset that is applied globally to all data samples, Local Feature Selection (LFS) finds optimal feature subsets for each local region of a data space. In this way, LFS is better able to adapt to regional variability and non-stationarity in a sample space. In addition, the method is paired with a simple classifier based on class similarity which can account for the fact that different samples may be modeled using different feature subsets. 

Local feature selection is performed by promoting class-wise clustering in the neighbourhood around each point, i.e., by finding the subset of available features that minimizes the average distance between points belonging to the same class, while maximizing the distances between classes. Thus, a feature space is identified that maximizes classifiability locally around each point. However, since this feature space can be different for each local region, standard classifiers cannot be readily applied. Instead, a notion of similarity between samples is introduced that intuitively lends itself to classification. Since LFS defines a local region for each sample, the regions are overlapping. Therefore, each point is represented in a number of feature spaces. Therefore, a class label can be assigned by accumulating the class labels of the nearest neighbours to a sample in each of these feature spaces. Full details of LFS, including experimental results demonstrating its effectiveness compared to other feature selection and classification methods on several datasets, are given in [@Armanfard:2015] and [@Armanfard:2017]. 

``LFSpy`` was designed to be used for researchers working on any supervised learning problem, but is especially powerful for data that are non-stationary, non-ergodic, or that otherwise do not cluster well into classes. A prominant example of data with these properties is electroencephalography (EEG) time-series, for which LFS has been used to continuously detect characteristic brain responses to auditory stimuli in coma patients [@Armanfard:2019].

# Usage
``LFSpy`` is a Python implementation of LFS that follows the ``scikit-learn`` [@scikit-learn] class structure, and can therefore be used as part of a ``scikit-learn`` Pipeline like other classifiers. Open-source code, full documentation, and a demo using sample data are available at https://github.com/McMasterRS/LFSpy.git. Using LFS to train a model and test on new data is made simple with `LFSpy` and can be done in just a few lines of code. Usage follows the standard format used for `scikit-learn` classifiers. First, an LFS object is created to hold the model configuration, parameters, and once trained, the trained model itself. The implementation is flexible in that it gives the user control over a number of optional parameters, including for example, the size of the local region used for feature selection. The following training and testing functions can then be called from that object:

 - `lfs.fit`: trains an LFS model given training data and corresponding training labels
 - `lfs.predict`: for a trained model, outputs class label predictions given testing data
 - `lfs.score`: outputs the classification error for the testing data in total, and by class, given testing data and ground truth testing labels

Given training and testing data that are compatible with `scikit-learn` models, a typical example of model training and testing is as follows:

    from LFSpy import LocalFeatureSelection
    lfs = LocalFeatureSelection(alpha=19, 
                                gamma=0.2, 
                                tau=2, 
                                sigma=1, 
                                n_beta=20, 
                                nrrp=2000, 
                                knn=1)
    lfs.fit(training_data, training_labels)
    predicted_labels = lfs.predict(testing_data)
    total_error, class_error = lfs.score(testing_data, testing_labels)

`LFSpy` is also fully compatible with the `scikit-learn` Pipeline method:

    from LFSpy import LocalFeatureSelection
    from sklearn.pipeline import Pipeline
    lfs = LocalFeatureSelection(alpha=19, 
                                gamma=0.2, 
                                tau=2, 
                                sigma=1, 
                                n_beta=20, 
                                nrrp=2000, 
                                knn=1)
    pipeline = Pipeline([('lfs', lfs)])
    pipeline.fit(training_data, training_labels)
    predicted_labels = pipeline.predict(testing_data)
    total_error, class_error = pipeline.score(testing_data, testing_labels)

<!---
The LFS method involves a number of parameters that are implemented in `LFSpy`. Their definitions are given as follows:

* `alpha`: the maximum number of selected features for each representative point
* `gamma`: impurity level tolerance, controls proportion of out-of-class samples can be in local region
* `tau`: number of passes through the training set
* `sigma`: adjusts weightings for observations based on their distance, values greater than 1 result in lower weighting
* `n_beta`: number of beta values to test
* `nrrp`: number of iterations for randomized rounding process
* `knn`: number of nearest neighbours to compare for classification
--->

The dependencies for `LFSpy` are as follows:

* Python 3
* [NumPy](https://numpy.org/)>=1.14
* [SciPy](https://www.scipy.org/)>=1.1
* [Scikit-learn](https://scikit-learn.org/stable/index.html)>=0.18.2

# Comparison to Other Classifiers
A comparison of classification accuracies obtained with LFS and two standard `scikit-learn` pipelines are shown below. The Random Forest classifier (RFC) and a linear Support Vector Machine (SVM) with univariate feature selection using the F-statistic are used for comparison. For all tests, we use default settings. For consistency, none of the methods are provided with a priori information about the number of informative features to select. Both LFS and RFC choose the number of appropriate features internally. The SVM must be given a number of features to choose, so we set the number of features to 25% of the total number of available features for this example.

Results are obtained with two sample datasets that are representative of the intended use case of LFS. The first is a sample dataset used to illustrate the utility of LFS. This dataset is synthetically generated with 100 training samples and 108 test samples. The number of informative vs. uninformative features in this dataset are unknown. The second dataset is the Iris dataset included with `scikit-learn`, which contains 100 samples and four features (50 are used for training, and 50 are used for testing; all four features are informative). To illustrate the value of LFS, we show the classification accuracy of each method after appending increasing numbers of up to 1000 non-informative Guassian features. Each feature was randomly generated with zero mean and a standard deviation between 0 and 3, sampled from a uniform distribution. 

It can be seen that with both datasets LFS outperforms the other two methods, particularly when the number of non-informative features becomes large. LFS remains relatively stable in classification performance, whereas RFC and SVM experience significant degradation as the number of non-informative features grows well past 100.

### Classification accuracies with the sample synthetic dataset

<!-- ![synth_data][results_sample] -->
![synth_data](https://github.com/McMasterRS/LFSpy/blob/master/LFSpy/comparisons/SampleData_Results.png)

### Classification accuracies with the Iris dataset

![iris_data][results_iris]

# Acknowledgments
Funding for this project was obtained through the CANARIE Research Software Program Local Support Initiative.

# References


[results_sample]: https://github.com/McMasterRS/LFSpy/blob/master/LFSpy/comparisons/SampleData_Results.png
[results_iris]: https://github.com/McMasterRS/LFSpy/blob/master/LFSpy/comparisons/IrisData_Results.png
