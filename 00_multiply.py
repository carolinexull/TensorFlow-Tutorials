import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32)
y = tf.multiply(x, x)

with tf.Session() as sess:
  #print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1, 1)
  print(sess.run(y, feed_dict={x: 1.0}))