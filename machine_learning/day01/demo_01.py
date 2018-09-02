import tensorflow as tf
import numpy as np

# a=tf.constant([1.0,2.0],name="a")
# b=tf.constant([2.0,3.0],name="b")
# result=a+b
# sess=tf.Session()
# print(sess.run(result))

# 拟合直线y=0.1x+0.3
# 创建数据
x_data = np.random.rand(100).astype(np.float32)  # 随机100个(0,1)的随机数据 是一个数组类型的数据
y_data = x_data * 0.1 + 0.3
print(y_data)
print("=" * 40)
# 搭建模型
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 一维 取值为-1 到 1 类型还是矩阵类型  并不是一个参数
biases = tf.Variable(tf.zeros([1]))  # 一维 矩阵类型并不是一个参数

y = Weights * x_data + biases
# 计算误差 y_data表示真实值  y表示预测值
loss = tf.reduce_mean(tf.square(y - y_data))  # 损失函数 tf.reduce_mean()函数:计算张量的各个维度上的元素的平均值。

# 传播误差
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 激活函数 0.5指的是学习率
train = optimizer.minimize(loss)  # 这句话的意思就是取要取损失函数的极小值

print(type(train))
#  开始训练
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好

sess = tf.Session()
sess.run(init)  # Very important
print(sess.run(y), "yyyyyyyyyyyyyyyy")
print("=" * 40)
print(sess.run(loss), "hahahhahahahaha")
# 训练200次
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
