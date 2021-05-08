Loss functions
===

cross entropy
---
* `sum(y_i log(p_i)) `
* `y_i` is the ground truth probability of i-th class and `p_i` is the predicted prob of i-th class
* when use sigmoid as activation, use cross entropy loss, instead of L2 loss
  * reason: cross entropy loss can help adjust "learning rate", fast when high deviation, slow when small devitation, unlike L2 loss

log-loss = binary cross entropy
---
* logistic regression (sigmoid as activation) uses log-loss as loss function

