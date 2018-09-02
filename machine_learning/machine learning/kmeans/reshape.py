import tensorflow as tf

a = [[1, 2], [3, 4]]

data = tf.equal(a, 1)
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

print(sess.run(tf.reshape(a, [1, -1])))
