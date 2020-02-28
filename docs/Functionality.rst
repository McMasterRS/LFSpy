Functionality
=============

::

    class LocalFeatureSelection(self, alpha=19, gamma=0.2, tau=2, sigma=1, n_beta=20, nrrp=2000, knn=1, rr_seed=None)

+----------------+-----------------------------------------------------------------------------+
| **Parameters** | **alpha : integer, optional, default 19**                                   |
|                |    maximum number of selected features for each representative sample       |
|                |                                                                             |
|                | **gamma : integer, optional, default 0.2**                                  |
|                |    impurity level                                                           |
|                |                                                                             |
|                | **tau : integer, optional, default 2**                                      |
|                |    number of iterations                                                     |
|                |                                                                             |
|                | **sigma : integer, optional, default 1**                                    |
|                |    controls neighboring samples weighting                                   |
|                |                                                                             |
|                | **n_beta : integer, optional, default 20**                                  |
|                |    number of distinct beta                                                  |
|                |                                                                             |
|                | **nrrp : integer, optional, default 2000**                                  |
|                |    number of iterations for randomized wandering process                    |
|                |                                                                             |
|                | **knn : integer, optional, default 1**                                      |
|                |    k nearest neighbours                                                     |
|                |                                                                             |
|                | **rr_seed : integer, optional, default None**                               |
|                |    seed value for random wandering process                                  |
|                |                                                                             |
+----------------+-----------------------------------------------------------------------------+
| **Attributes** | **fstar : array of shape (n_features, n_features)**                         |
|                |    selected features for each sample                                        |
|                |                                                                             |
|                | **fstar_lin : array of shape (n_features, n_features)**                     |
|                |    fstar before applying randomized wandering process                       |
|                |                                                                             |
|                | **training_data : array of shape (n_features, n_samples**                   |
|                |    The set of M by N features and observations the model was trained on     |
|                |                                                                             |
|                | **training_labels : array of shape (n_samples)**                            |
|                |    The set of N class labels the model was trained on                       |
|                |                                                                             |
+----------------+-----------------------------------------------------------------------------+

Methods
-------

+-----------------------------------------------------+-----------------------+
| fit(self, training_data, training_labels)           |                       |
+-----------------------------------------------------+-----------------------+
| predict(self, testing_data)                         |                       |
+-----------------------------------------------------+-----------------------+
| classification(self, testing_data)                  |                       |
+-----------------------------------------------------+-----------------------+
| class_sim_m(self, test, N, patterns, targets, fstar)|                       |
+-----------------------------------------------------+-----------------------+

::

    __init__(self, alpha=19, gamma=0.2, tau=2, sigma=1, n_beta=20, nrrp=2000, knn=1, rr_seed=None)

Initialize self

::

    fit(self, training_data, training_labels)

Fit model

+----------------+------------------------------------------------------------------+
| **Parameters** | training_data : {array-like} of shape (n_samples, m_features)    |
|                |        Training data                                             |
|                |                                                                  |
|                | training_labels : {array-like} of shape (n_samples)              |
|                |        Class labels for each sample                              |
|                |                                                                  |
+----------------+------------------------------------------------------------------+
| **Returns**    |                                                                  |
+----------------+------------------------------------------------------------------+

::

    predict(self, testing_data)

Predict using the model

+----------------+------------------------------------------------------------------+
| **Parameters** | testing_data : {array-like} of shape (n_samples, m_features)     |
|                |        Testing data                                              |
|                |                                                                  |
+----------------+------------------------------------------------------------------+
| **Returns**    |                                                                  |
+----------------+------------------------------------------------------------------+

::

    classification(self, testing_data)

Internal feature classification function, called by predict function

+----------------+------------------------------------------------------------------+
| **Parameters** | testing_data : {array-like} of shape (n_samples, m_features)     |
|                |        Testing data                                              |
|                |                                                                  |
+----------------+------------------------------------------------------------------+
| **Returns**    |                                                                  |
+----------------+------------------------------------------------------------------+


::

    class_sim_m(self, test, N, patterns, targets, fstar, gamma, knn)

Internal feature classification function, called by classification function

+----------------+------------------------------------------------------------------+
| **Parameters** | test : {array-like} of shape (n_samples, m_features)             |
|                |        Testing data                                              |
|                |                                                                  |
|                | N: {integer}                                                     |
|                |        Number of features                                        |
|                |                                                                  |
|                | patterns:                                                        |
|                |       Data the model was trained on                              |
|                |                                                                  |
|                | targets:                                                         |
|                |      Class Labels the model was trained on                       |
|                |                                                                  |
|                | fstar:                                                           |
|                |      Selected features for each samples                          |
|                |                                                                  |
|                | gamma:                                                           |
|                |      Impurity Level                                              |
|                |                                                                  |
|                | knn:                                                             |
|                |      K nearest neighbours                                        |
|                |                                                                  |
+----------------+------------------------------------------------------------------+
| **Returns**    |                                                                  |
+----------------+------------------------------------------------------------------+
