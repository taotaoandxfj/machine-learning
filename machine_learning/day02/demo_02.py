import tensorflow as tf

# 声明w1,w2两个变量。这里还通过seed设定了随机种子
# 这样可以保证每次运行的结果时一样的
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常亮，注意这里x时一个1X2的矩阵
# x = tf.constant([[0.7, 0.9]])

# 通过3.4.2节描述的前向传播算法获得神经网络的输出
# 这里可以定义placeholder作为存放数据的地方。这里的维度也不一定要定义
# 但如果维度时确定的，那么给出维度可以降低出错的概率
# x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
# 其实我感觉啊  这个是占位置用的感觉一样  其实开始的时候是么有数据的.
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")  # 3x2

a = tf.matmul(x, w1)  # 矩阵乘法
y = tf.matmul(a, w2)  # 矩阵乘法

sess = tf.Session()

# 与之前的计算不同，这里不能直接通过sess.run(y)来获取y的取值
# 因为w1和w2都还没有运行初始化过程,也就是初始化方法并没有真正的运行。以下两行分别初始化w1和w2的两个变量
# sess.run(w1.initializer)
# sess.run(w2.initializer)

init_op = tf.global_variables_initializer()  # 初始化所有的变量
sess.run(init_op)
# 需要指定字典
# print(sess.run(y))
print("=" * 40)
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
sess.close()
