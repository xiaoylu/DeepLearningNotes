The bias-variance tradeoff 
===

one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data.
However, these two goals can not be fulfilled at the same time.

* dimensionality reduction and feature selection can decrease variance by simplifying models. 
* a larger training set tends to decrease variance. 
* adding features (predictors) tends to decrease bias, at the expense of introducing additional variance.
* boosting combines many "weak" (high bias) models in an ensemble that has lower bias than the individual models, while bagging combines "strong" learners in a way that reduces their variance.
