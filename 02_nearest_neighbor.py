import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)


Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

xtr = tf.placeholder("float", [None, 784])
xte =  tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

pred = tf.arg_min(distance, 0)

accuracy = 0.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        nn_idx = sess.run(pred, feed_dict={xtr:Xtr, xte:Xte[i, :]})
        print "text", i, "predict:", np.argmax (Ytr[nn_idx]), "true class:", np.argmax(Ytr[nn_idx])

        # Calculate accuracy
        if np.argmax(Ytr[nn_idx]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
