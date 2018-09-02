import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf

'''
    使用tensorflow实现k-means算法

'''

# 随机生成数据,且将数据分成两个类
num_puntons = 2000
conjunto_puntos = []  # 数据存放在列表中去

for i in range(num_puntons):
    if np.random.random() < 0.5:
        conjunto_puntos.append([np.random.normal(0., 0.9), np.random.normal(0., 0.9)])  # np.random.normal()表示正态分布

    else:
        conjunto_puntos.append([np.random.normal(3., 0.5), np.random.normal(3., 0.5)])

df = pd.DataFrame({"x": [v[0] for v in conjunto_puntos],
                   "y": [v[1] for v in conjunto_puntos]
                   })

# sns.lmplot('x', 'y', data=df, fit_reg=False, size=6)
#
# plt.show()

vectors = tf.constant(conjunto_puntos)

k = 2  # 聚成4个类

# 随机取四个类的中心
'''
    下面的这句代码的解释:
    目的:选取随机生成的2000个点中的四个点作为聚类中心
    函数解释:tf.random_shffle() 将vetors进行打乱 以达到随机生成的目的
    tf.slice():参考https://blog.csdn.net/chenxieyy/article/details/53031943
    或者本目录中的slice函数

'''
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

# 维度的增加 目的是为了后面的减法(tf.subtract)如果放在二维平面上进行减法会报类型大小不匹配的错误 所以必须升维
expand_vectors = tf.expand_dims(vectors, 0)
expand_centroides = tf.expand_dims(centroides, 1)  # 这个1表示变成1行2列4高  具体自己打印出来体会即可

diff = tf.subtract(expand_vectors, expand_centroides)  # 对应位置的减法运算

sqr = tf.square(diff)  # 平方

distance = tf.reduce_sum(sqr, 2)  # 求和

# 选取每一个数据点距离最近聚类中心的点(返回的时最小值的索引)
assignments = tf.argmin(distance, 0)  # [0 1 1 ..., 2 0 3]
'''
    下面的这句代码的解释:
    c=[0,1,2,3]
    tf.where（）返回bool型tensor中为True的位置
    tf.equal()函数表示相等就返回true 否则返回false 但是注意如果传入的是两个列表的话 那么必须两个列表的长度是相等的
    tf.gather():表示用第二个参数中所表示的索引从 第一个参数中取出相应的元素组成新的列表 可参考https://blog.csdn.net/guotong1988/article/details/53172882
    tf.concat()链接函数.0表示纵向链接
'''
means = tf.concat([tf.reduce_mean(tf.gather(vectors,
                                            tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
                                  reduction_indices=[1]) for c in range(k)], 0)

# tf.assign()函数表示将第二个参数的值更新给第一个参数
update_centroides = tf.assign(centroides, means)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for step in range(100):
    _, centroide_values, assignment_values = sess.run([update_centroides, centroides, assignments])

data = {'x': [], 'y': [], 'cluster': []}

for i in range(len(assignment_values)):
    data['x'].append(conjunto_puntos[i][0])
    data['y'].append(conjunto_puntos[i][1])
    data['cluster'].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot('x', 'y', data=df, fit_reg=False, size=6, hue='cluster', legend=False)
plt.show()
# print(conjunto_puntos[0])
# print(conjunto_puntos[2])
# print(assignment_values)
# print(centroide_values)
'''
    k-means模型的评价标准:有均一性,完整性,V-measure,轮廓系数,ARI等
    在这里选用轮廓系数作为评价的标准
    样本i的轮廓系数:s(i)
    簇内不相似度:计算样本i到同簇其它样本的平均距离为a(i),应尽可能小。
    簇间不相似度:计算样本i到其它簇C j 的所有样本的平均距离b ij ,应尽可能大。
    轮廓系数s(i)值越接近 1 表示样本i聚类越合理,越接近-1,表示样本i应该分类到另外
    205机器学习课程-第 8 周-聚类(Clustering)
    的簇中,近似为 0,表示样本i应该在边界上;所有样本的s(i)的均值被成为聚类结果的轮廓系
    数。
    s(i) =(b(i) − a(i))/max{a(i), b(i)}

'''
print("=" * 10)
# 点集合的分开
data = []
for c in range(k):
    # data1 = tf.where(tf.equal(assignment_values, c))
    data.append(tf.reshape(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignment_values, c)), [1, -1])), [-1, 2]))
    # data2 = tf.reshape(data, [-1, 2])
    # data = tf.gather(vectors,tf.where(tf.equal(assignment_values, c)))
# data1 = tf.where(tf.equal(assignment_values, 0))
# data3 = tf.reshape(data1, [1, -1])
# print(sess.run(data1))
# 求同类中一个点到其他点之间的平均距离
x = data[0][0]
# 升维方便做减法
expand_data0 = tf.expand_dims(data[0], 0)
expand_data1 = tf.expand_dims(data[1], 0)

expand_x = tf.expand_dims(tf.expand_dims(x, 0), 1)
# print(sess.run(expand_data0))
# print(sess.run(expand_x))
# sub = tf.subtract(expand_data0, expand_x)
# square = tf.square(sub)
reduce_sum1 = tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(expand_data0, expand_x)), 2))
reduce_sum2 = tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(expand_data1, expand_x)), 2))

print(sess.run(reduce_sum1))
print(sess.run(reduce_sum2))
print(sess.run((reduce_sum2 - reduce_sum1) / (tf.maximum(reduce_sum1, reduce_sum2))))

# print(sess.run((tf.shape(expand_vectprint(sess.run(data2))ors))))
# print(sess.run((tf.shape(expand_centroides))))
# print(vectors)
