import tensorflow as tf

a = [True, False, True, False, True, True]

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

print(sess.run(tf.reshape(tf.where(a), [1, -1])))
