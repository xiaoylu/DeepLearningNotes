Deep Learning Notes
===

Loss Functions
---
* Log loss: the cross-entropy between prediction `y` and truth `Y` is `- y*np.log(Y) - (1-y)*np.log(1-Y)`
* tanh: `np.tanh(x)`
* Softmax: stablize it by making the exponent non-positive.
```
def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)
```
