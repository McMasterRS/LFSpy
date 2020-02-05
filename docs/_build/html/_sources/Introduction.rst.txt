Introduction
=================================

Localized feature selection (LFS) is a supervised machine learning approach for embedding localized feature selection in classification. The sample space is partitioned into overlapping regions, and subsets of features are selected that are optimal for classification within each local region. As the size and membership of the feature subsets can vary across regions, LFS is able to adapt to local variation across the entire sample space.

This repository contains a python implementation of this method that is compatible with scikit-learn pipelines. For a Matlab version, refer to https://github.com/armanfn/LFS

The LFS approach was developed by Nargus Armanfard. For further information please refer to:

* N. Armanfard, JP. Reilly, and M. Komeili, "Local Feature Selection for Data Classification", IEEE Trans. on Pattern Analysis and Machine Intelligence, vol. 38, no. 6, pp. 1217-1227, 2016.
* N. Armanfard, JP. Reilly, and M. Komeili, "Logistic Localized Modeling of the Sample Space for Feature Selection and Classification", IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 5, pp. 1396-1413, 2018

Statement of Need
---------------------------------

LFSpy offers an implementation of the Local Feature Selection (LFS) algorithm that is compatible with scikit-learn, one of the most widely used machine learning packages today. LFS combines classification with feature selection, and distinguishes itself by it flexibility in selecting a different subset of features for different data points based on what is most discriminative in local regions of the feature space. This means LFS overcomes a well-known weakness of many classification algorithms, i.e., classification for non-stationary data where the number of features is high relative to the number of samples. 
