# 完整的神经网络样例程序
# 下面给出一个完整的程序来训练神经网络解决二分类的问题
import tensorflow as tf
# 通过numpy工具包生成模拟数据集
from numpy.random import RandomState

# 定义数据集batch的大小
batch_size = 8  # 反向传播算法的第一步
# 定义神经网络的参数，这里还是沿用之前的结构
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))
# 在shape的一个维度上使用None可以方便的使用不同的batch大小。在训练数据需要把分
# 成比较小的batch,但在测试的时候,可以一次性使用全部的数据.当数据集比较小的时候这样比较方便测试,但
# 数据集比较大的时候,将大量的数据放入一个batch可能会导致内存溢出
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")  # 这个None很精髓 嘻嘻嘻
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义神经网络的前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法
y = tf.sigmoid(y)  # 好像是激活函数的意思啊兄弟 原来时二分类问题中 可以将一个线性变成非线性的样子
cross_entry = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))  # 将y限制到0.1 到 1.0的范围内
    + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))  # 交叉熵
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entry)

# 通过随机数来生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本的标签.在这里所有x1+x2<1的样例都被认为时正样本(比如零件合格),
# 而其他为负样本(比如零件不合格),和tensorflow游乐场中的表示方法不太一样的时
# 在这里使用过0表示负样本,1来表示正样本.大部分解决分类问题的神经网络都会采用0和1表示法

Y = [int(x1 + x2 < 1) for (x1, x2) in X]

# 创建一个会话来运行tensorflow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))
    '''
    在训练之前神经网络参数的值
    w1=[[-0.81131822  1.48459876  0.06532937]
        [-2.4427042   0.0992484   0.59122431]]
    w2=[[-0.81131822]
        [ 1.48459876]
        [ 0.06532937]]
    '''
    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次训练batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step,
                 feed_dict={x: X[start, end], y_: Y[start, end]}
                 )
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(
                cross_entry, feed_dict={x: X, y_: Y}
            )
            print("After %d training step(s) ,cross entropy on all data is %g" % (i, total_cross_entropy))
        # 打印训练之后神经网络参数的值
        print("=" * 40)
        print(sess.run(w1))
        print(sess.run(w2))
