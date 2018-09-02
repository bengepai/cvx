import numpy as np
from random import shuffle

def svm_loss_naive(W,X,y,reg):
    """
    SVM损失函数，带循环的naive的实现方式
    :param W:
    :param X:
    :param y:
    :param reg:
    :return:
    """
    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
        if j == y[i]:
            continue
        margin = scores[j] - correct_class_score + 1
        if margin > 0:
            loss += margin
            dW[:, j] += X[i].T
            dW[:, y[i]] += -X[i].T

    loss /= num_train
    dW /= num_train

    loss += 0.5*reg*np.sum(W*W)
    dW += reg*W