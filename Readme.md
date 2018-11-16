Deep Learning Notes
===

Important Functions
---
* Log loss: the cross-entropy between prediction `y` and truth `Y`. Loss function used in neural networks which have softmax activations in the output layer (because of convex optimization?). Deritative = `g ( 1 - g )` where `g` is the log-loss.

```
def logloss(y, Y):
    return - y*np.log(Y) - (1-y)*np.log(1-Y)

def dlogloss(y, Y):
    g = logloss(y, Y)
    return 
```

* tanh: `np.tanh(x)` better than log-loss in some cases. Deritative = `1 - g*g ` where `g` is the tanh.

* Softmax: stablize it by making the exponent non-positive.
```
def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)
```
