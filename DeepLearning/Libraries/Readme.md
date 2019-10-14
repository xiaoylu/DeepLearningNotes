Tensorflow
---

[v2.0 Guide](https://github.com/tensorflow/docs/tree/master/site/en/r1/guide)
 
One builds a computational graph `tf.Graph` by defining `tf.Tensor` and `tf.Operation` and run the computational graph with `tf.Session`. During a call to `tf.Session.run`, any `tf.Tensor` only has a single value.



Comparison between Tensorflow and PyTorch
---

* Dynamic Graph: Tensorflow has pre-defined structure, while PyTorch can change the structure by inputs. Tensorflow RNN deals with different length of input (like different #words in a sentence) by padding, PyTorch unrolls LTSM units.
* to be cont.
