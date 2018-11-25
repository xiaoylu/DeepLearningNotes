Ensemble Methods
===

Ensemble methods combine the predictions of several base estimators:

* Averaging method: build estimators independently and average their predictions, such as random forests
* Boosting method: build estimators sequentially to reduce the bias of previous estimators, such as gradient boosted trees

Decision Trees:
---
* Select the attribute which brings the most information gain to split samples, i.e. the one with max Kullback–Leibler divergence. (Or other impurity measures such as Gini, misclassifications etc.)

> `InformationGain(Y, x_i) = H(Y) - H(Y|x_i)`
where `H()` is the entropy/conditional entropy. In short, decision trees maximize the amount of information gained about the prediction `Y` from observing that attribute `x_i`.

* When used for feature selection: 
importance is calculated by the amount that each attribute split point improves the performance measure, 
weighted by the number of observations the node is responsible for.

* A tree with few samples and many attributes is very likely to overfit:
    * External
        * getting the right ratio of samples to number of features
        * performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand
        * sampling an equal number of samples from each class to balance the dataset 
    * Internal regularization:
        * limit max. depth of trees
        * ensembles / bag more than just 1 tree
        * set stricter stopping criterion on when to split a node further
  
