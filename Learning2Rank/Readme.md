Learning to Rank
===

Categories:
* Pointwise:

* Pair-wise: the learning task is formalized as classification of object pairs into two categories (correctly ranked and incorrectly ranked).
  * Models:
    * Rank SVM
    * RankNet (see below for paper review)
  * Pros:
    * Extract ground truth from users’ clicks-through data
    * 
  * Cons:

* List-wise: 













Review of some key papers
===

RankNet - "Learning to rank using gradient descent by microsoft research (ICML 2005)"
---
The learning algorithm is given a set of pairs {(a, b)} that each pair is labeled `P_{ab}`, 
i.e. the posterior probability that a is ranked higher than b
* Dataset: {(a, b)}, `P_{ab}` 
* Features: each a or b has many features one can extract from its content (Document keywords etc.)

The **cross entropy** cost function is applied to measure the error of prediction `P_{ij}`.

And the key simplification is that
```
          exp(o_ij)
P_ij = ---------------
        1 + exp(o_ij)
```
where the "distance" between i and j is defined as
```
o_ij = o_i - o_j
```
Then the authors show that it is sufficient to train with only **adjacent** samples in a ranked list, 
because they can uniquely identify `P_ij` for any pair (i, j). This step reduces the computational time significantly.

The neutral network outputs o_i directly. But we have no idea about what o_i should be. Only relations are what we have, i.e. `P_{ab}`

Thus, given the `P_{ab}` between a and b, the training algorithm works on a cost function cost(o_i, o_j). 

According to the chain rule the derivative of cost over neural network's weight would propagate as
```
d  cost         d cost
--------  = --------------- * (dcost/dweight - dcost/dweight) 
d weight     d (o_i - o_j)
```

So it works similarly as the back-prop, even though we do not know the ground truth for its output `o_i`.
