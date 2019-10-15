Tensorflow
---

[Guide](https://github.com/tensorflow/docs/tree/master/site/en/r1/guide)
 
Tensorflow is a library for building and running computational graphs `tf.Graph` which comprised of `tf.Tensor`s (tensors) and `tf.Operation`s ("ops"). 

A call to `tf.Session.run` would specify the values of any `tf.Tensor` in a graph. `tf.Tensor` class has `eval()` method while `tf.Operation` class has `run()` method. Tensors contains `tf.Variable`, `tf.constant`, `tf.placeholder` and `tf.SparseTensor`.

`tf.layers` are the preferred way to add **trainable parameters** to a graph. 
The layer infers the number of its internal variables by inspecting the input. 
But the output size must be specified using the `units` argument.

```python
# v1.0
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
  # train is an op, loss is a tensor
  # train has no value, loss has value
  _, loss_value = sess.run((train, loss))  
  print(loss_value)

print(sess.run(y_pred))
```

Comparison between Tensorflow and PyTorch
---

* Dynamic Graph: Tensorflow has pre-defined structure, while PyTorch can change the structure by inputs. Tensorflow RNN deals with different length of input (like different #words in a sentence) by padding, PyTorch unrolls LTSM units.
* to be cont.
