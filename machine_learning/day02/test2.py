import tensorflow as tf
import numpy as np


# 只有一层神经网络结构 inputs为300x1的矩阵 weights为1x10的矩阵
def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 建立一个in_size*out_size的矩阵  随机数
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 在这里不能直接相加0
    # tf.add(biases, 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    print(type(wx_plus_b), "xixixixixixixxi")
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


# 使用numpy完成对数据的构建
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # linspace方法创建一个从-1到1的300个数的等差数列 矩阵的形式为300x1

noise = np.random.normal(0, 0.05, x_data.shape)

y_data = np.square(x_data) - 0.5 + noise  # 对等差数列中的每一个数做平方 其实说白了 这里还是拟合函数x*x-0.5+noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义影藏层 有两层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  # 指的是激励函数
predition = add_layer(l1, 10, 1, activation_function=None)

# 计算预测值与真实值的差距 平均值

# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition)), reduction_indices=[1])
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                                    reduction_indices=[1]))

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predition),
#                                               reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 0,1表示学习效率

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(5000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

print("=" * 40)
print(sess.run(predition, feed_dict={xs: x_data, ys: y_data}))
