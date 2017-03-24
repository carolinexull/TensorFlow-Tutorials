import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

lr = 0.01
epochs = 100
display_step = 50

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name="W")
b = tf.Variable(rng.randn(), name="b")

pred = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for(x, y) in zip(train_X, train_Y):
            sess.run(train_op, feed_dict={X:x, Y:y})

        if(epoch+1)%display_step==0:
            c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print ("Epoch:", '%04d'%(epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print "optimization Finished"
    training_cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
    print "training cost:", training_cost, "W=", sess.run(W), "b=", sess.run(b)

    plt.plot(train_X, train_Y, 'ro', label="original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted Line")
    plt.legend()
    plt.show()

    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(tf.reduce_sum(tf.pow(pred-Y, 2))/(2*test_X.shape[0]), feed_dict={X:test_X, Y:test_Y})
    print "testing cost:", testing_cost

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()