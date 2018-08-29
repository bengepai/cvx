import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass
    def train(self,X,y):
        """
        训练分类器，对于KNN来说，这只是存储训练数据

        :param X: 训练图片矩阵
        :param y: 图片矩阵对应的标签
        :return:
        """
        self.X_train = X
        self.Y_train = y

    def predict(self,X,k=1,num_loops=0):
        """
        用该分类器预测测试数据的标签
        :param X: 测试图片矩阵
        :param k: 选取的临近点的数量
        :param num_loops: 决定用哪种方式来计算距离
        :return: 预测的标签
        """
        if num_loops == 0 :
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invaild value')
        return self.predict_labels(dists,k=k)

    def compute_distances_two_loops(self,X):
        """
        计算测试X与X_train中每个数据的距离
        :param X:测试数据
        :return:距离矩阵
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum(np.square(X[i]-self.X_train[j])))
        return dists

    def compute_distances_one_loop(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train))
        for i in range(num_test):
            dists[i] = np.sum(np.square(self.X_train - X[i]),axis = 1)
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train))

        dists = -2*np.dot(X,self.X_train.T)
        sq1 = np.transpose([np.sum(np.square(X),axis=1)])
        sq2 = np.sum(np.square(self.X_train),axis=1)
        dists = np.add(dists,sq1)
        dists = np.add(dists,sq2)
        dists = np.sqrt(dists)

        return dists


    def predict_labels(self,dists,k=1):
        """
        通过距离矩阵和训练数据集来预测测试数据的label
        :param self:
        :param dist:
        :param k:
        :return:
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.Y_train[dists[i].argsort()[0:k]]
            closest_y = [int(i) for i in closest_y]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred