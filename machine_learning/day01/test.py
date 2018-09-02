import tensorflow as tf
import numpy as np

x = np.random.rand(100).astype(np.float32)
y = x * 0.3 + 0.1
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
a = tf.constant([[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]])
init = tf.global_variables_initializer()
a_average = tf.reduce_mean(a, 0)
sess = tf.Session()
sess.run(init)
# print(sess.run(Weights))
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
print(x_data)
# print(x)
# print(type(x))
# print(y)
# print(sess.run(a))
# print(sess.run(a_average))
