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
        correct_class_score = scores[int(y[i])]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, int(y[i])] += -X[i].T

    loss /= num_train
    dW /= num_train

    loss += 0.5*reg*np.sum(W*W)
    dW += reg*W


    return loss, dW

def svm_loss_vectorized(W,X,y,reg):
    loss = 0.0
    dW = np.zeros(W.shape)

    scores = X.dot(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    y = [int(val) for val in y]
    correct_class_scores = scores[range(num_train),list((y))].reshape(-1,1)
    margins = np.maximum(0,scores - correct_class_scores + 1)
    margins[range(num_train),list(y)] = 0
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W*W)

    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

    dW = (X.T).dot(coeff_mat)
    dW = dW/num_train + reg*W

    return loss, dW
