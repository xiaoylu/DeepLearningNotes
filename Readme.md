Deep Learning Notes
===

Important Functions
---
* Softmax: stablize it by making the exponent non-positive.
```
# Input: row vector X of shape (n, 1)
# Output: row vector p of shape (n, 1), each component in range [0, 1]
def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)
```
During backpropagation, we pass to the previous layer the derivative of outputs `p` over inputs `X` (see [math](https://deepnotes.io/softmax-crossentropy)) 
```
# Input: row vector X of shape (n, 1)
# Output: derivative of output p[i] over input X[j]
def pass(X, i, j):
    p = stable_softmax(X)
    return p[i] * (1 - p[j]) if i == j else - p[j] * p[i]
```

* Log loss: the cross-entropy between prediction `p` and truth `y`, it is used in logistic regression as the loss function
```
def logloss(p, y):
    return - y*np.log(p) - (1-y)*np.log(1-p)
```
It is also commonly used as loss function in neural networks which have softmax activations in the output layer (because of convex optimization?)
```
def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1) indicating the class of each example (the y-th neural output should be 1, all other outputs should be 0)
    """
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss
```

* logistic function/sigmoid function: `g = 1 / ( 1 + np.exp(-x))`. Deritative = `g * (1 - g)`.

* tanh: `g = np.tanh(x)` better than logistic as loss function in some cases. Deritative = `1 - g * g`.

* ReLU (Rectified Linear Unit): `g = max(x, 0)`. Why ReLU?
    * Easy to get derivative: efficient 
    * Avoid vanishing gradient problem: it saturates to only one direction; unlike tanh and sigmoid which saturate to both direction because their `dg/dx -> 0` when `x -> -inf` or `x -> inf`.
    
Vanishing Gradient
---
Backpropagation after many layers, the gradients become close to 0, rarely change the first `k` input layers.
Solution:
* ResNet: use ensembles to make sure the layers are actually shallow.
* LSTM: for RNN.
* ReLU: one direction saturation
* multilevel hierarchy: unsuperivsed pre-training of each layers



Acknowledgement:
* [https://deepnotes.io/softmax-crossentropy](https://deepnotes.io/softmax-crossentropy)
* [https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/1-%20Neural%20Networks%20and%20Deep%20Learning](https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/1-%20Neural%20Networks%20and%20Deep%20Learning)
