import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    softmax 损失函数，用循环的naive实现
    输入数据维度是D，有C个类，我们在N个例子的minibatch中进行操作。
    :param W: 维度(D,C)
    :param X: 维度(N,D)
    :param y: 维度(N,)
    :param reg: 正则化系数
    :return:
    """
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        shift_scores = scores - max(scores)
        loss_i = -shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
        loss += loss_i
        for j in range(num_classes):
            softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_output) * X[i]
            else:
                dW[:, j] += softmax_output * X[i]
    loss /= num_train
    loss += 0.5*reg*np.sum(W*W)
    dW = dW/num_train + reg*W

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    loss = 0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    shift_scores = scores - np.max(scores, axis=1).reshape(-1,1)
    sofmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
    loss = -np.sum(np.log(sofmax_output[range(num_train),list(y)]))
    loss /= num_train
    loss += 0.5*reg*np.sum(W*W)

    dS = sofmax_output.copy()
    dS[range(num_train),list(y)] += -1
    dW = (X.T).dot(dS)
    dW = dW/num_train + reg*W

    return loss, dW