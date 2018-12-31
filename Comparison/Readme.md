Comparison Questions
===
Acknowledgement: the list of comparison questions is originally collected by [Shujian Liu](https://www.linkedin.com/in/shujian-liu/).
Source: https://www.linkedin.com/pulse/ml-2-shujian-liu/

L1 and L2 regularization: 
---
https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when

L1 encourages sparsity of the weight vector (Lasso) while L2 penalizes large weight dramatically (due to the sqr in Ridge). Elastic Net is a good choice.

Logistic Regression vs Naive Bayes
---
https://www.quora.com/What-is-the-difference-between-logistic-regression-and-Naive-Bayes
[Short review of Logistic Regression](https://drive.google.com/file/d/1LSAQsQndQUmLO45NOMSwlFAvy1XGfhqH/view)

* Navie Bayes is a Generative model while Logistic Regression is a Discriminative model
* Navie Bayes assume naive conditional independence that `P(y|x1,x2..xn) = P(y|x1) P(y|x2) .. P(y|xn)`; LR allows correlation between features.
* Naive Bayes is fast estimating `P(y)*P(y|x1,x2..xn)` by Maximum A Posteriori (MAP); LR requires training by grad descent
* LR can overfit when #features >> #samples; Naive Bayes work fine due to its simplicity
* Priors improve Naive Bayes; Regularization improve Logistic Regression

Linear Regression vs Logistic Regression
---
https://stats.stackexchange.com/questions/29325/what-is-the-difference-between-linear-regression-and-logistic-regression/29326#29326

Logistic regression falls into the form of genelarized linear model (GLM) that `P(Y) = sigmoid(AX + b)` where `sigmoid-1()` is the link function `link_function(P(Y)) = linear_regression(X)`. Note that in GLM, the output is continuous while LR outputs a prob. `P(Y)`.

LR vs SVM: 
---
https://stats.stackexchange.com/questions/95340/comparing-svm-and-logistic-regression



SVM, dual vs primal
---
https://www.quora.com/Why-is-solving-in-the-dual-easier-than-solving-in-the-primal-What-advantages-do-we-get-from-solving-in-the-dual 

LDA vs. NB: https://www.quora.com/Classification-machine-learning-What-are-the-main-differences-between-the-LDA-Linear-Discriminant-Analysis-and-Naive-Bayes-classifiers

LDA vs. PCA: https://www.quora.com/What-is-the-difference-between-LDA-and-PCA-for-dimension-reduction

Gradient Boosting Tree vs Random Forest: https://stats.stackexchange.com/questions/173390/gradient-boosting-tree-vs-random-forest

K-means and hierarchical clustering: https://www.quora.com/What-is-the-difference-between-k-means-and-hierarchical-clustering

Generative vs. discriminative models: https://stats.stackexchange.com/questions/12421/generative-vs-discriminative

RNN vs. CNN (at high level): https://datascience.stackexchange.com/questions/11619/rnn-vs-cnn-at-a-high-level

Word2vec vs. GloVe (word embeddings): https://www.quora.com/How-is-GloVe-different-from-word2vec

