import tensorflow as tf
from numpy.random import RandomState

# #### 1. 定义神经网络的相关参数和变量。
batch_size = 8
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# #### 2. 设置自定义的损失函数。
# 定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测。
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# #### 3. 生成模拟数据集。
rdm = RandomState(1)
dateset_size = 128
X = rdm.rand(dateset_size, 2)
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# #### 4. 训练模型。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        # start = (i * batch_size) % dateset_size
        # end = (i * batch_size) % dateset_size + batch_size
        start = (i * batch_size) % dateset_size
        end = min(start + batch_size, dateset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            print("After %d training step(s), w1 is: " % (i))
            # print(sess.run(w1))
    print(sess.run(w1))
