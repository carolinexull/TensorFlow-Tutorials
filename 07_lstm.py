import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

input_vec_size = lstm_size = 28
time_step_size = 28

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W, B, lstm_size):
    #X:[batch_size, time_step_size, input_vec_size]
    #XT:[time_step_size, batch_size, input_vec_size]
    XT = tf.transpose(X, [1, 0, 2])
    #lstm_size = input_vec_size.
    XR = tf.reshape(XT, [-1, lstm_size])

    X_split = tf.split(XR, time_step_size, 0)

    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    output, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)

    return tf.matmul(output[-1], W) + B, lstm.state_size

mnist = input_data.read_data_sets("/home/liuchong/PycharmProjects/tensorflow/Fun-Try/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

X = tf.placeholder(tf.float32, [None, 28, 28])
Y = tf.placeholder(tf.float32, [None, 10])

W = init_weights([lstm_size, 10])
B = init_weights([10])

py_x, state_size = model(X, W, B, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X:trX[start:end], Y:trY[start:end]})
        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print (i, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X:teX[test_indices]})))