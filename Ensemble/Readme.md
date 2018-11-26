Ensemble Methods
===

Ensemble methods combine the predictions of several base estimators:

* Averaging method: build estimators independently and average their predictions to reduce total variance, such as random forests
* Boosting method: build weak estimators sequentially to reduce the bias of previous estimators, such as AdaBoost, gradient boosted trees

Boosting is based on weak learners (high bias, low variance). In terms of decision trees, weak learners are shallow trees. Boosting reduces error mainly by reducing bias (and also to some extent variance, by aggregating the output from many models).

On the other hand, Random Forest (with bagging/averaging) uses fully grown decision trees (low bias, high variance). It tackles the error reduction task in the opposite way: by reducing variance. The trees are made uncorrelated to maximize the decrease in variance, but the algorithm cannot reduce bias (which is slightly higher than the bias of an individual tree in the forest). Hence the need for large, unpruned trees, so that the bias is initially as low as possible.

Please note that unlike Boosting (which is sequential), RF grows trees in parallel.

Decision Trees:
---
* Select the attribute which brings the most information gain to split samples, i.e. the one with max Kullback–Leibler divergence. (Or other impurity measures such as Gini, misclassifications etc.)

> `InformationGain(Y, x_i) = H(Y) - H(Y|x_i)`
where `H()` is the entropy/conditional entropy. In short, decision trees maximize the amount of information gained about the prediction `Y` from observing that attribute `x_i`.

* When used for feature selection: 
importance is calculated by the amount that each attribute split point improves the performance measure, 
weighted by the number of observations the node is responsible for.

* A tree with few samples and many attributes is very likely to overfit:
    * External data preparation
        * getting the right ratio of samples to number of features
        * performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand
        * sampling an equal number of samples from each class to balance the dataset 
    * Internal regularization:
        * limit max. depth of trees
        * ensembles / bag more than just 1 tree
        * set stricter stopping criterion on when to split a node further
        
* As neighboring data points are more likely to lie within the same leaf of a tree, the RF can perform an implicit, non-parametric density estimation, and it can transform the data into another [feature space](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py).

* Decision trees have a number of abilities that make them valuable for boosting, namely the ability to handle data of mixed type and the ability to model complex functions.

AdaBoost
---
* fit a sequence of weak learners on repeatedly modified versions of the data
* each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones

Gradient Boosting Regression Tree (GBRT)
---

* Same idea, use steepest descent to greedily improve predictions.
* Regularization:
   * Subsampling: a random sub-set of samples to train the next tree
   * Shrinkage: learning rate to decay the importance of latter trees
* xgboost used a more regularized model formalization than GBM to control over-fitting, which gives it better performance.



