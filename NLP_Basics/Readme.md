NLP Basics
===

Word Represenation
===

CBOW (context word -> center word) vs SkipGram (word2vec, center word -> context word)
---
https://cs224d.stanford.edu/lecture_notes/notes1.pdf
* Compared to traditional matrix factorization, new input words can be added dynamically
* CBOW is learning to predict the word by the its context, skip-gram model is designed to predict the context
* Skip-gram: works well with small amount of the training data, represents well even rare words or phrases
* CBOW: several times faster to train than the skip-gram, slightly better accuracy for the frequent words

Named Entity Recognition (NER)
---
Instead of feeding a word vector, feeding a concat window vector (apple has different meaning in different context)

Sequential models
---
* RNN, (bi-)LSTM, GRU
  * one hidden state  
  * slowly than parallel models

Attention models
---
* Transformer
  * self dot attention: softmax(Q^T K) V
  * multi-head attention
  * Add&Norm: residual 
  * Feed forward
  * Positional coding 
  * Masks
* BERT
  *  Pretrain: 
    * hide k% input words, predict hidden words in the output layer
    * predict two sentence order  
