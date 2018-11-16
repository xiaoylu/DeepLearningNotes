Deep Learning Notes
===

Important Functions
---
* Log loss: the cross-entropy between prediction `y` and truth `p`. Loss function used in neural networks which have softmax activations in the output layer (because of convex optimization?). Deritative = `g ( 1 - g )` where `g` is the log-loss.

```
def logloss(y, p):
    return - y*np.log(p) - (1-y)*np.log(1-p)

def dlogloss(y, p):
    g = logloss(y, p)
    return 
```

* tanh: `np.tanh(x)` better than log-loss in some cases. Deritative = `1 - g*g ` where `g` is the tanh.

* Softmax: stablize it by making the exponent non-positive.
```
def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)
```
