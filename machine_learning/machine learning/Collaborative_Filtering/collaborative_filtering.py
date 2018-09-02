import numpy as np
import pandas as pd
import tensorflow as tf


# 数据处理
def data_handing():
    # 加载ratings.csv文件
    ratings_df = pd.read_csv('ratings.csv')
    # print(ratings_df.head(3))

    # 加载movies.csv文件
    movies_df = pd.read_csv('movies.csv')
    # print(movies_df.head(3))

    # 给movies_df增加一列movieRow
    movies_df['movieRow'] = movies_df.index
    # print(movies_df.head(3))

    # 筛选movies_df中的特征 只包括movieRow movieId title
    movies_df = movies_df[['movieRow', 'movieId', 'title']]
    # print(movies_df.head(3))

    # 将帅选后的特征存储到一个moviesProcessed.csv中
    movies_df.to_csv('moviesProcessed.csv', index=False, header=True, encoding='utf-8')

    # 将ratings_df 和 movieId合并
    ratings_df = pd.merge(ratings_df, movies_df, on='movieId')

    # print(ratings_df)
    # ratings_df的筛选,只要userId,movieRow,rating这三列
    ratings_df = ratings_df[['userId', 'movieRow', 'rating']]

    # 将筛选后的数据存储到一个ratingsProccessed.csv中
    ratings_df.to_csv("ratingsProccessed.csv", index=False, header=True, encoding='utf-8')

    # 创建电影评分矩阵rating 和评分记录矩阵record
    userNo = ratings_df['userId'].max() + 1  # 最大用户编号
    # print(userNo)
    movieNo = ratings_df['movieRow'].max() + 1  # 最大电影编号
    # print(movieNo)
    # rating矩阵的建立
    rating = np.zeros((movieNo, userNo))

    # print(rating)
    ratings_df_length = np.shape(ratings_df)[0]  # 求行数
    for index, row in ratings_df.iterrows():
        rating[int(row['movieRow']), int(row['userId'])] = row['rating']

    # print(rating)

    record = rating > 0
    # print(record)
    record = np.array(record, dtype=int)
    # print(record)
    return rating, record


# 模型的构建:均值归一化 其实这里有个bug就是如果用户的评分就是0咋办?
'''
    说一下为啥要均值归一化的原因:因为在创建rating矩阵中有很多的0,这个0代表的就是没有进行评分的用户
    虽然用0表我觉的并不是非常的合理,因为有些用户的评分可能就是0,在下面的我会就此的进行改进.所以需要进行
    均值归一化,详情可以看吴恩达机器学习笔记16.6

'''
'''
    模型的训练

'''


def normalRatings(rating, record):
    m, n = rating.shape  # m:电影的数量;n:用户数量
    rating_mean = np.zeros((m, 1))  # 每一行的均值
    # rating_norm = np.zeros((m, n))  # 原始评分减去均值  我觉得这里博主写错了
    rating_norm = rating
    for i in range(m):
        idx = record[i, :] != 0  # 注意这里的返回的时true or false的形式
        # print(idx)
        rating_mean[i] = np.mean(rating[i, idx])  # 计算每一行的平均值
        rating_norm[i, idx] -= rating_mean[i]

    rating_mean = np.nan_to_num(rating_mean)
    rating_norm = np.nan_to_num(rating_norm)

    # print(rating_mean)
    return rating_norm, rating_mean


def train(rating_norm, rating_mean, rating):
    # 假设有10种类型的电影
    num_features = 10
    m, n = rating_norm.shape
    # 初始化电影内容矩阵x和用户喜爱矩阵theta,标准正态分布
    x_parameters = tf.Variable(tf.random_normal([m, num_features], stddev=0.35))
    theta_parameters = tf.Variable(tf.random_normal([n, num_features], stddev=0.35))
    # 构建损失函数J(theta) 详细的数学形式见吴恩达笔记16.4
    # 梯度下降
    # 下面的矩阵相乘函数tf.matmul()中的transpose_b表示在进行相乘之间进行倒置.,置换
    global_step = tf.Variable(0)
    # 学习率的设置 指数衰减 原文这里设置的学习率是固定的0.0001这样会导致梯度下降的特别的慢
    # 所有并不推荐这样做 解决方法是通过学习率指数衰减俩进行解决.意思就是刚开始的学习率高下降的快,后面的
    # 的时候学习率下降的慢,这是一种很好的加快梯度下降的方式
    '''
        下面介绍一下各个参数的意思:
        0.5:代表初始学习率
        global-step:代表学习的当前步骤
        100:表示没过100轮迭代乘0.95
        staircase:为True表示没100步更新学习率  为False表示每一步都更新学习率
        
    '''
    learning_rate = tf.train.exponential_decay(0.01, global_step, 500, 0.96, staircase=True)
    loss = 1 / 2 * tf.reduce_sum(
        ((tf.matmul(x_parameters, theta_parameters,
                    transpose_b=True) - rating_norm) * record)) ** 2 + 1 / 2 * tf.reduce_sum(
        x_parameters ** 2) + 1 / 2 * tf.reduce_sum(theta_parameters ** 2)
    # 目标的优化
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 将结果显示到tensorboard中去
    tf.summary.scalar('loss', loss)
    summaryMerged = tf.summary.merge_all()  # 自动管理
    filename = './movie_tensorboard'
    writer = tf.summary.FileWriter(filename)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(6000):
            _, movie_summary = sess.run([train, summaryMerged])
            # sess.run(train)
            global_step = i
            writer.add_summary(movie_summary, i)
            if i % 100 == 0:
                print(sess.run(loss))  # 240.063
        # 将当前的x与theta进行保存
        current_x_parameters, current_theta_paramaters = sess.run([x_parameters, theta_parameters])
        print("current_x_parameters:", current_x_parameters)
        print("current_theta_paramaters:", current_theta_paramaters)
        predicts = np.dot(current_x_parameters, current_theta_paramaters.T) + rating_mean  # 结果预测
        errors = np.sqrt(np.sum((predicts - rating) ** 2))  # 误差
        print("errors is :", errors)
        return predicts


def film_recommend(predicts):
    user_id = input("您要向哪位用户进行电影推荐呢?请输入用户的id:")
    sortedResult = predicts[:, int(user_id)].argsort()[::-1]  # 降序
    idx = 0  # 用来表示向用户推荐的电影数量 标志结束位
    print("为用户推荐的评分最高的20部电影是:".center(80, '='))
    movie_df = pd.read_csv("moviesProcessed.csv")
    for i in sortedResult:
        print('评分:%2f,电影名:%s' % (predicts[i, int(user_id)], movie_df.iloc[i]['title']))
        idx += 1
        if idx % 20 == 0:
            break


if __name__ == '__main__':
    rating, record = data_handing()
    rating_norm, rating_mean = normalRatings(rating, record)
    predicts = train(rating_norm, rating_mean, rating)
    film_recommend(predicts)
    # print(rating)
    # print(record)
