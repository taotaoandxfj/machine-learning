import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)  # 乘法

with tf.Session() as sess:
    ressult = sess.run(output, feed_dict={input1: [7.0], input2: [2.0]})
    print(ressult)
 