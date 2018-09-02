import tensorflow as tf
import numpy as np

# 使用tensorflow来拟合直线y=0.3*x+1
# 1.首先创建相关的数据

x_data = np.random.rand(100).astype(np.float32)
y_data = 0.3 * x_data + 0.1

# 2.使用tensorflow来构造数据

weights = tf.Variable(tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float32))
biases = tf.Variable(tf.zeros([1]))

# 预测数据
y_pre = weights * x_data + biases

# 3.构造损失函数 并进行梯度下降算法
loss = tf.reduce_mean(tf.square(y_pre - y_data))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


#用



# 4.进行训练
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(201):
        sess.run(train)
        if i % 10 == 0:
            print("weights:", sess.run(weights), "   ", "biases:", sess.run(biases))
