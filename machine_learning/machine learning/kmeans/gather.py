import tensorflow as tf

a = [[1, 2], [3, 4], [4, 5]]
b = tf.constant(a)
init = tf.global_variables_initializer()
sess = tf.Session()
c = [[0, 1]]
d = tf.constant(c)
sess.run(init)
print(sess.run(b))
print(sess.run(d))

print(sess.run(tf.gather(b, d)))
