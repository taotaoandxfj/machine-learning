import tensorflow as tf

a = tf.constant([[1, 2], [3, 4], [5, 6]])
b = tf.constant([[1, 2]])
a1 = tf.expand_dims(a, 0)
b1 = tf.expand_dims(b, 1)
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
print(sess.run(tf.shape(a1)))
print(sess.run(tf.shape(b1)))
print("=" * 10)
print(sess.run(a1))
print(sess.run(b1))
print("=" * 10)
data = tf.subtract(a1, b1)
print(sess.run(tf.subtract(a1, b1)))

print(sess.run(tf.shape(data)))

print("=" * 10)

sqr = tf.square(data)

distance = tf.reduce_sum(sqr, 2)

min = tf.argmin(distance, 0)
print(sess.run(distance))
print("=" * 10)
print(sess.run(min))  # [0 1 1] 这儿返回的是最小值的索引

