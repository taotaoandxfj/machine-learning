# 利用神经网络进行函数的拟合
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def interfence(x_input, in_size, out_size, activate_function=None):
    # global weights
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(x_input, weights) + biases
    if activate_function == None:
        output = Wx_plus_b
    else:
        output = activate_function(Wx_plus_b)
    return output


# def interfence(inputs, in_size, out_size, activation_function=None):
#     weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 建立一个in_size*out_size的矩阵  随机数
#     biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 在这里不能直接相加0
#     # tf.add(biases, 0.1)
#     wx_plus_b = tf.matmul(inputs, weights) + biases
#     print(type(wx_plus_b), "xixixixixixixxi")
#     if activation_function is None:
#         outputs = wx_plus_b
#     else:
#         outputs = activation_function(wx_plus_b)
#     return outputs


# 构造数据进行拟合 这是真实值哈
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 300x1
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 占位置张量

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 第一层隐藏层
l1 = interfence(xs, 1, 10, activate_function=tf.nn.relu)
# 第二层影藏层
prediction = interfence(l1, 10, 1, activate_function=None)

# 损失函数的定义与计算
# 并没有使用正则化可能会出现过拟合现象
loss = tf.reduce_mean(
    tf.reduce_sum(tf.square(ys - prediction),
                  reduction_indices=[1]))  # + tf.contrib.layers.l2_regularizer(0.02)(weights)

# 优化梯度下降
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 声明
init = tf.global_variables_initializer()

# 用可视化工具画出拟合的过程
fig = plt.figure()  # 先生成一个图片框
ax = fig.add_subplot(1, 1, 1)  # 增加一个编号
ax.scatter(x_data, y_data)
plt.ion()

# 开始训练
with tf.Session() as sess:
    sess.run(init)

    for i in range(1001):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            # 用线将其拟合出来
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.05)
plt.ioff()
plt.show()
