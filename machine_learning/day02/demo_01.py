import tensorflow as tf

# 声明w1,w2两个变量。这里还通过seed设定了随机种子 这儿有两层
# 这样可以保证每次运行的结果时一样的
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))  # 2x3
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))  # 3x1

# 暂时将输入的特征向量定义为一个常亮，注意这里x时一个1X2的矩阵
x = tf.constant([[0.7, 0.9]])  # 1x2

# 通过3.4.2节描述的前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)  # 矩阵乘法
y = tf.matmul(a, w2)  # 矩阵乘法

sess = tf.Session()

# 与之前的计算不同，这里不能直接通过sess.run(y)来获取y的取值
# 因为w1和w2都还没有运行初始化过程,也就是初始化方法并没有真正的运行。以下两行分别初始化w1和w2的两个变量
# sess.run(w1.initializer)
# sess.run(w2.initializer)

init_op = tf.global_variables_initializer()  # 初始化所有的变量
sess.run(init_op)

print(sess.run(y))
sess.close()
