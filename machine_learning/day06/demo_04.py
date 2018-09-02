import tensorflow as tf

# 以下的代码给出了如何通过变量重名民来直接读取变量的滑动平均值

v = tf.Variable(0, dtype=tf.float32, name="v")
# 通过变量名重命名将原来的变量v的滑动平均值直接复赋值给v
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "model2/model2.ckpt")
    print(sess.run(v))  # 0.0999999 这个值就是原来模型中变量v的滑动平均值
