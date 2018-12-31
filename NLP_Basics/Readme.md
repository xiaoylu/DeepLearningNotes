NLP Basics
===

Word Represenation
===

CBOW vs SkipGram
---
https://cs224d.stanford.edu/lecture_notes/notes1.pdf
* Compared to traditional matrix factorization, new input words can be added dynamically
* CBOW is learning to predict the word by the its context, skip-gram model is designed to predict the context
* Skip-gram: works well with small amount of the training data, represents well even rare words or phrases
* CBOW: several times faster to train than the skip-gram, slightly better accuracy for the frequent words
