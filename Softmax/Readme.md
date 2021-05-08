Softmax
===

One softmax for all
---
* Multi-class **single-label** classification: Softmax assumes that each example is a member of exactly one class.
* Multi-class multi-label requires multiple sigmoids

Candidate Sampling
---
* Only compute a subset of logits and backpropogate along these logits.
* And sampling softmax: a faster way to train a softmax classifier
* https://www.tensorflow.org/extras/candidate_sampling.pdf
