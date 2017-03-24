import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print "add with constant :", sess.run(a+b)
    print "multi with constant:", sess.run(a*b)

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print sess.run(add, feed_dict={a:2, b:3})
    print sess.run(mul, feed_dict={a:2, b:3})

mat1 = tf.constant([[2., 2.]])
mat2 = tf.constant([[3.], [3.]])

product = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    print sess.run(product)

