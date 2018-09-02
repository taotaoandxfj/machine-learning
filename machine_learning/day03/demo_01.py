import tensorflow as tf

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

with tf.Session() as sess:
    print(tf.clip_by_value(v, 2.5, 4.5).eval())  # [[ 2.5  2.5  3. ] [ 4.   4.5  4.5]]

    # tf.log函数可以完成对张量中所有的元素的依次求对数
    print("=" * 40)
    print(tf.log(v).eval())  # 计算log，一个输入计算e的ln，两输入以第二输入为底  tf.log(x, name=None)

    # 求tensor平均值
    print("=" * 40)
    print(tf.reduce_mean(v).eval())  # 3.5
'''
在这里总结一下:损失函数的使用
如果遇到是分类问题损失函数用softmax+corss_entry(交叉熵可以解决)
上述两个合在一起的函数是tf.nn.softmax_cross_entroy_with_logits(labels=y_,logits=y)
交叉熵的公式是祥见p75

如果遇到是回归问题损失函数用均方差来表达(MSE) 公式详见p78



'''
