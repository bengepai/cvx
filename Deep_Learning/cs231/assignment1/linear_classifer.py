import numpy as np
from linear_svm import *
from softmax import *

class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """
        用随机梯度下降算法来训练SGD
        :param X: 维度为(N,D),有N个维度为D的训练数据
        :param y: 维度为(N,),有N个训练标签
        :param learning_rate: 优化的学习速率
        :param reg: 正则化系数
        :param num_iters: 优化的迭代次数
        :param batch_size: 每一步使用的训练样例个数
        :param verbose:
        :return: 包含每一次迭代损失函数值的列表
        """

        num_train, dim = X.shape
        num_classes = int(np.max(y) + 1)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            loss,grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W += -learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history

    def predict(self, X):
        """
        使用线性分类器训练好的权重来预测数据点的标签
        :param X:
        :return:
        """
        y_pred = np.zeros(X.shape[1])

        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        计算损失函数以及他的导数
        :param X_batch:
        :param y_batch:
        :param reg:
        :return:
        """
        pass

class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)