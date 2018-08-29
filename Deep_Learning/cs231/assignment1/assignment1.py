from k_nearest_neighbor import KNearestNeighbor
from load_CIFAR10 import load_CIFAR10
import random
import os
import numpy as np
import matplotlib.pyplot as plt

def picture_sample_show():
    classes = ['airplane','car','bird','cat','deer','dog','frog','horse']
    num_classes=  len(classes)
    samples_per_class = 10
    for y,cls in enumerate(classes):
        idxs = np.flatnonzero(Y_train == y)
        idxs = np.random.choice(idxs,samples_per_class,replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class,num_classes,plt_idx)
            plt.imshow(X_train[idx].reshape(3,32,32).transpose(1,2,0).astype("float").astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

# 获取数据
X_train,Y_train,X_test,Y_test = load_CIFAR10("D:\\gitCode\\CVX_Code\\Deep_Learning\\cs231\\assignment1\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py")
#picture_sample_show()

# 取出子集
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
Y_train = Y_train[mask]

num_test = 500
X_test = X_test[:num_test]
Y_test = Y_test[:num_test]


num_folds = 5
k_choices = [1,3,5,8,10,12,15,20,50,100]

x_train_folds = []
y_train_folds = []

x_train_folds = np.array_split(X_train,num_folds)
y_train_folds = np.array_split(Y_train,num_folds)


k_to_accuracies = {}

classifier = KNearestNeighbor()
for k in k_choices:
    accuracies = np.zeros(num_folds)
    for fold in range(num_folds):
        temp_X = x_train_folds[:]
        temp_y = y_train_folds[:]
        x_validate_fold = temp_X.pop(fold)
        y_validate_fold = temp_y.pop(fold)

        temp_X = np.array([y for x in temp_X for y in x])
        temp_y = np.array([y for x in temp_y for y in x])
        classifier.train(temp_X,temp_y)

        y_test_pred = classifier.predict(x_validate_fold, k=k)
        num_correct = np.sum(y_test_pred == y_validate_fold)
        accuracy = float(num_correct) / num_test
        accuracies[fold] = accuracy
    k_to_accuracies[k] = accuracies

for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k,accuracy))