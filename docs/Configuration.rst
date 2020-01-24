Configuration
=================================

The localized feature selection method has a set of user configurable paramaters that can be tweaked to get your desired functionality. For a full description of each parameter refer to the paperslisted in the cirations section. The parameters are:

* alpha: (default: 19) the maximum number of selected features for each representative point

* gamma: (default: 0.2) impurity level tolerance, controls proportion of out-of-class samples can be in local region

* tau: (default: 2) number of passes through the training set

* sigma: (default: 1) adjusts weightings for observations based on their distance, values greater than 1 result in lower weighting

* n_beta: (default: 20) number of beta values to test, controls the relative weighting of intra-class vs. inter-class distance in the objective function

* nrrp: (default: 2000) number of iterations for randomized rounding process

* knn: (default: 1) number of nearest neighbours to compare for classification