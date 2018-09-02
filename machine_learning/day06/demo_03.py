import tensorflow as tf

# 以下的代码给出了一个保存滑动平均模型的样例
v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有申明滑动平均模型时只有一个变量v,所以以下的语句只会输出"v:0"
for variables in tf.global_variables():
    print(variables.name)
# 加载滑动平均模型

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
# 在申明滑动平均模型以后,TensorFlow会自动生成一个影子变量
# 以下的语句会输出"v:0"和"v/ExponentialMovingAverage:0"
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存的时候,TensorFlow会将v:0和v/ExponentialMovingAverage:0这两个变量都保存下来
    saver.save(sess, "model2/model2.ckpt")
    print(sess.run(v))
