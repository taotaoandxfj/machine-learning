import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
MNIST数字识别问题

'''

# MNIST数据集中相关的参数
INPUT_NODE = 784  # 输入层的节点数.对于mnist数据集来说就是图片的像素
OUTPUT_NODE = 10  # 输出层的节点数  用数组的形式来进行表示如[0,0,1,0,0,0,0,0,0,0]表示的就是2 以此类推

# 配置神经网络参数
LAYER1_NODE = 500  # 隐藏层节点数.这里使用只有一个隐藏层的网络结构作为样例
BATCH_SIZE = 100  # 一个训练batch中训练数据个数.数字越小的时候,训练过程越接近,随机梯度下降,数字越大的时候,训练越接近梯度下降

LEARNING_RATE_BASE = 0.8  # 基础的学习率,也就是最开始的学习率
LEARNING_RATE_DECAY = 0.99  # 衰减率,也就是最开始的衰减率

REGULARIZATION_RATA = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRANING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 这是 一个辅助函数.给定神经网络的输入和所有的参数,功能是计算神经网络的前向传播结果 这儿有两层
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有使用滑动平均类时,直接使用当前参数的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果,这里直接使用了ReLU激活函数.
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)  # 第一层的输出结果
        layer2 = tf.matmul(layer1, weights2) + biases2  # 第二层的输出结果
        return layer2
    else:
        # 首先使用avg_class.average函数来计算的出变量的滑动平均值
        # 然后再计算相应的神经网络前向传播的结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        layer2 = tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
        return layer2


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')  # 输入 INPUT_NODE=784
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')  # 输出 OUTPUT_NODE=10
    # 生成隐藏层的参数
    # tf.truncated_normal()函数:从截断的正态分布中输出随机值。类似于tf.random_normal()函数
    # 生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
    # tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

    # shape: 一维的张量，也是输出的张量。
    # mean: 正态分布的均值。
    # stddev: 正态分布的标准差。
    # dtype: 输出的类型。
    # seed: 一个整数，当设置之后，每次生成的随机数都一样。
    # name: 操作的名字。

    # weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    weight1 = tf.Variable(tf.random_normal([INPUT_NODE, LAYER1_NODE]))

    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))  # 生成一个一位数组长度为LAYER1_NODE 每一个的值为0.1

    # 生成输出层的参数
    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))  # 生成一个一位数组长度为OUTPUT_NODE 每一个的值为0.1

    # 计算当前参数下神经网络前向传播的结果,这里给出的用于计算滑动平均的类为None
    # 所以函数不会使用参数的滑动平均值
    y = inference(x, None, weight1, biases1, weight2, biases2)  # y指的是预测结果

    # 定义存储训练的轮数的变量.这个变量不需要计算滑动平均值,所以这里指定这个变量为
    # 不可训练变量(trianable=False).在使用tensorflow训练神经网络时,
    # 一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 给定,滑动平均率和训练轮数的变量,初始化滑动平均类,目的是为了加快早期的衰减速率 variable_averages表示模型变量
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)  # global_step是为了控制衰减率二设置的变量

    # 在代表所有的神经网络参数的变量上使用滑动平均....tf.trainable_variables()返回的就是图上集合中的元素

    # 这里定义了一个更新滑动变量的操作,这里需要给定一个列表,每次执行这个操作的时候,这个列表中的变量都会被更新

    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 通过一个影子变量来记录滑动平均值
    average_y = inference(x, variable_averages, weight1, biases1, weight2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间的差距的损失函数
    # sparse_softmax_cross_entropy_with_logits()这个函数的第一个参数表示神经网络中不包含softmax层的前向传播结果,
    # 第二个参数表示训练数据的正确答案
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 这里使用L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATA)
    # 计算模型的正则化损失,一般只计算边上权重的正则化损失,而不是用偏置项
    regularization = regularizer(weight1) + regularizer(weight2)
    # 总损失等于交叉熵损失和正则化损失
    loss = cross_entropy_mean + regularization

    # 设置损失函数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础的学习率,随着迭代的进行,学习率会逐渐的递减
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的数据需要迭代的次数
        LEARNING_RATE_DECAY  # 学习衰减速度,初始的衰减率
    )

    # 使用优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 这里测操作时通过反向传播函数来更新神经网络中的参数,且更新每一个参数的滑动平均值
    train_op = tf.group(train_step, variables_averages_op)
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 下面的这个运算首先将一个布尔型的数值转换为实数型,然后计算平均值,这个平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程:
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 通过验证数据来大致判断停止的条件和判断训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # z准备验证数据,在真实的应用中,这部分数据在训练中时不可见的,这个数据只是为了模型的优劣做最后的评价
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代的训练神经网络
        for i in range(TRANING_STEPS):
            if i % 1000 == 0:
                validata_acc = sess.run(accuracy, feed_dict=validate_feed)
                print(i, validata_acc)

            # 产生这一轮使用一个batch的训练数据,并运行整个训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        # 在训练结束后,在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("最终模型的准确率是", test_acc)


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
