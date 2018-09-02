import tensorflow as tf

a = [1, 3, 2, 1, 3, 1, 0]
b = [0, 1, 2, 3, 4]

data = tf.equal(a, 1)
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

print(sess.run(data))
