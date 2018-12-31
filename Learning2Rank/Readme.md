Learning to Rank
===

Categories
---

* Pointwise: a scoring system which output a score based on the feature vector
  * Global order directly

* Pair-wise: the learning task is formalized as classification of object pairs into two categories (correctly ranked and incorrectly ranked).
  * Models:
    * RankSVM
    * RankBoost
    * RankNet (see below for paper review)
  * Pros:
    * easy ground truth extraction from users’ clicks-through data
  * Cons:
    * quadratic #pairs for training

* List-wise: treat a rank list as the label directly, optimizing MAP or NDCG
   * Models:
      * LambdaRank
      * AdaRank
      * SoftRank
      * LambdaMART
   * Score:
      * MAP (Mean Average Precision)
      * NDCG (Normalized Discounted Cumulative Gain)

Review of some key papers
===

RankNet - "Learning to rank using gradient descent by microsoft research (ICML 2005)"
---
The learning algorithm is given a set of pairs {(a, b)} that each pair is labeled `P_{ab}`, 
i.e. the posterior probability that a is ranked higher than b
* Input: 
  * {(a, b)}, `P_{ab}` 
  * Features: each a or b has many features one can extract from its content (Document keywords etc.)
* Output:
  * Given a new pair {(a, b)} output the prob a ranked higher than b, `P_{ab}`.

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
