import numpy as np
from sklearn.decomposition import PCA
import sys

'''
    唯维度的选择:k的选择
'''


def index_list(lst, componet=0, rate=0):
    if componet and rate:
        print("componet and rate must choose only one ")
        sys.exit(0)
    if not componet and not rate:
        print('Invalid parameter for numbers of components!')
        sys.exit(0)
    elif componet:
        print('Choosing by component, components are %s......' % componet)
        return componet
    else:
        print("choosing by rate ,and it is %s ...." % rate)
        '''
            解释一下下面的代码:这里对维度选择首选是从k=1开始逐渐的增加直到满足平均均方误差与训练方差的值小于等于(1-rate)
            也就是奇异矩阵的前k维向量与奇异举证前n个向量的比值需要大于等于rate
            详细的过程可以参考吴恩达笔记14.5章有较为详细的讲解
        '''
        for i in range(1, len(lst)):  # lst指的是s--奇异矩阵
            if sum(lst[:i]) / sum(lst) >= rate:
                return i  # 这个返回的i就是k即需要降到的维度
        return 0


def pca_by_svd():
    mat = [[-1, -1, 0, 2, 1], [2, 0, 0, -1, -1], [2, 0, 1, 1, 0]]
    Mat = np.array(mat, dtype=np.float64)
    print("before PCA transforMation data is \n", Mat)
    p, n = np.shape(Mat)  # 3*5
    t = np.mean(Mat, 0)  # 计算样本每个特征值的平均值
    print(t)
    # 每个样本数据减去改特征的平均值(归一化处理)
    for i in range(p):
        for j in range(n):
            Mat[i, j] = float(Mat[i, j] - t[j])
    # print(Mat)
    # 求协方差矩阵
    cov_Mat = np.dot(Mat.T, Mat) / (p - 1)

    # pca by numpy.svd :计算协方差的特征向量通过svd
    # 这里的u指的是一个n*n的矩阵,它是一个具有数据之间最小投射误差的方向向量构成的矩阵
    # 如果我们需要将数据从n维降至k维,我们只需要从u中选取前k个向量
    # 奇异值分解 其中s指的就是奇异矩阵 奇异值矩阵的平方对角线元素等于协方差矩阵的特征值
    u, s, v = np.linalg.svd(cov_Mat)
    index = index_list(s, rate=0.99)  # index指的就是k
    # Mat:3*5
    # u:5*5 --> 5*2
    T2 = np.dot(Mat, u[:, :index])
    print('We choose %d main factors.' % index)
    print('After PCA transformation, data becomes:\n', T2)

    '''
        解释一下这里的恢复:
        因为前面的得到新特征向量T=Mat*u[:,:index],这里的Mat相当于旧特征向量
        则由矩阵的关系可以到Mat1=T*U[:,:index].T
        其实这里的Mat和Mat1时是基本的的相等的,也不是完全的相等
    '''

    print('After PCA restore, data becomes:\n', np.dot(T2, u[:, :index].T))  # 矩阵的恢复
    '''
        method2:pca by sklearn
    
    '''
    pca = PCA(n_components=2)
    pca.fit(mat)
    print('Method 2: PCA by Scikit-learn:')
    print(pca.fit_transform(mat))


if __name__ == "__main__":
    pca_by_svd()
