import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/liuchong/PycharmProjects/tensorflow/Fun-Try/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

def model(X, w):
    return tf.matmul(X, w)

w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X:trX[start:end], Y:trY[start:end]})
        print (i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X:teX})))