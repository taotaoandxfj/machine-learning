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

'''

    对demo_01.py的inference函数做一个处理
    

'''


# 这是 一个辅助函数.给定神经网络的输入和所有的参数,功能是计算神经网络的前向传播结果 这儿有两层
def inference(input_tensor, reuse=False):
    # 根据传进来的reuse来判断是创建新变量还是使用自己创建好的.在第一次构造网络时
    # 需要创建新的变量,以后每次调用这个函数都直接使用reuse=True就不需要每次将变量传递进来了
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.truncated_normal_initializer(0, 0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # 第二层
    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.truncated_normal_initializer(0, 0))
        layer2 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    return layer2



