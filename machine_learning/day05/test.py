import tensorflow as tf

a = tf.constant(0.1, shape=[500])
b = tf.Variable(tf.zeros([1, 10]) + 0.1)
c = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))  # 0.1表示标准差

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
