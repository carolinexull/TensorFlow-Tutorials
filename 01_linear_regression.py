import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 101)
trY = 2*trX + np.random.randn(*trX.shape)*0.33

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

def model(X, w):
    return tf.multiply(X, w)

w = tf.Variable(0.0, name="W")
y_model = model(X, w)

cost = tf.square(Y - y_model)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        for (x, y) in zip (trX, trY):
            _, aa = sess.run([train_op, cost], feed_dict={X:x, Y:y})
        print aa

    print (sess.run(w))