import tensorflow as tf

v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

sess = tf.InteractiveSession()

# 验证greater()函数的作用
print(tf.greater(v1, v2).eval())  # [False False  True  True]
# 验证函数tf.where()函数的作用
print(tf.where(tf.greater(v1, v2), v1, v2).eval())  # [ 4.  3.  3.  4.]
sess.close()