from k_nearest_neighbor import KNearestNeighbor
from load_CIFAR10 import load_CIFAR10
from linear_classifer import LinearSVM
from linear_svm import svm_loss_naive,svm_loss_vectorized
from gradient_check import grad_check_sparse
import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt

X_train,Y_train,X_test,Y_test = load_CIFAR10("D:\\gitCode\\CVX_Code\\Deep_Learning\\cs231\\assignment1\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py")



num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

mask = range(num_training,num_training + num_validation)
X_val = X_train[mask]
Y_val = Y_train[mask]

mask = range(num_training)
X_train = X_train[mask]
Y_train = Y_train[mask]

#replace 为 false 代表选取的样本中没有重复值
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
Y_dev = Y_train[mask]

mask = range(num_test)
X_test = X_test[mask]
Y_test = Y_test[mask]

# 预处理，减去图像的平均值
mean_image = np.mean(X_train,axis=0)
#print(mean_image.shape)
#print(mean_image[:10])
#plt.figure(figsize=(4,4))
#plt.imshow(mean_image.reshape(3,32,32).transpose(1,2,0).astype('uint8'))
#plt.show()

X_train = np.hstack([X_train, np.ones((X_train.shape[0],1))])
X_val = np.hstack([X_val,np.ones((X_val.shape[0],1))])
X_test = np.hstack([X_test,np.ones((X_test.shape[0],1))])
X_dev = np.hstack([X_dev,np.ones((X_dev.shape[0],1))])

#print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)
W = np.random.randn(3073,10)*0.0001

#loss, grad = svm_loss_naive(W,X_dev,Y_dev,0.000005)
#f = lambda w: svm_loss_naive(w, X_dev, Y_dev, 0.0)[0]
#grad_numerical = grad_check_sparse(f,W,grad)
#print('turn on reg')
#loss, grad = svm_loss_naive(W, X_dev,Y_dev,5e1)
#f = lambda w:svm_loss_naive(w,X_dev,Y_dev,5e1)[0]
#grad_numerical = grad_check_sparse(f,W,grad)

#tic = time.time()
#loss_naive, grad_naive = svm_loss_naive(W, X_dev, Y_dev, 0.000005)
#toc = time.time()
#print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

#tic = time.time()
#loss_vectorized, _ = svm_loss_vectorized(W, X_dev, Y_dev, 0.000005)
#toc = time.time()
#print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

#print('difference: %f' % (loss_naive - loss_vectorized))

svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, Y_train, learning_rate=1e-7, reg=2.5e4, num_iters=1500, verbose=True)
toc = time.time()
print('That took %fs' % (toc-tic))