Introduction
=================================

Conventional feature selection algorithms assign a
single common feature set to all regions of the sample space.
In contrast, this paper proposes a novel algorithm for localized
feature selection for which each region of the sample space is
characterized by its individual distinct feature subset that may
vary in size and membership. This approach can therefore select
an optimal feature subset that adapts to local variations of the
sample space, and hence offer the potential for improved performance.
Feature subsets are computed by choosing an optimal
coordinate space so that, within a localized region, within-class
distances and between-class distances are, respectively, minimized
and maximized. Distances are measured using a logistic function
metric within the corresponding region. This enables the optimization
process to focus on a localized region within the sample
space. A local classification approach is utilized for measuring
the similarity of a new input data point to each class. The
proposed logistic localized feature selection (lLFS) algorithm is
invariant to the underlying probability distribution of the data;
hence, it is appropriate when the data are distributed on a
nonlinear or disjoint manifold. lLFS is efficiently formulated as a
joint convex/increasing quasi-convex optimization problem with
a unique global optimum point. The method is most applicable
when the number of available training samples is small. The
performance of the proposed localized method is successfully
demonstrated on a large variety of data sets.We demonstrate that
the number of features selected by the lLFS method saturates
at the number of available discriminative features. In addition,
we have shown that the Vapnik–Chervonenkis dimension of the
localized classifier is finite. Both these factors suggest that the
lLFS method is insensitive to the overfitting issue, relative to
other methods.