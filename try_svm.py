#!/usr/bin/env python
# -*- coding: utf-8 -*-


from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
import sys


"""
# TODO デバッグ
train_x = pca.fit_transform(train_x)
"""

def make_data(images):
    data = []
    for image in images:
        img = Image.open(image)
        img = np.asarray(img)
        shape = img.shape
        img = img.reshape(1, shape[0]*shape[1]*shape[2])[0]
        data.append(img)
    return np.asarray(data)


def pca(data):
    pass



def main(args):
    query = args
    img_dir = './images/'
    images = [img_dir + fname for fname in os.listdir(img_dir) if fname.endswith('jpg')]
    labels = [query[0] if query[0] in img.split('/')[-1] else query[1] for img in images]
    data = make_data(images)
    is_train = np.random.uniform(0, 1, len(data)) <= 0.7
    is_chick = np.where(np.array(labels) == query[0], 1, 0)
    train_x, train_y = data[is_train], is_chick[is_train]
    pca = PCA(n_components=5, svd_solver='randomized')
    train_x = pca.fit_transform(train_x)
    estimator = LinearSVC(C=1.0)
    estimator.fit(train_x, train_y)
    test_x, test_y = data[is_train == False], is_chick[is_train == False]
    test_x = pca.transform(test_x)
    confusion_matrix(test_y, estimator.predict(test_x))
    accuracy = accuracy_score(test_y, estimator.predict(test_x))
    print(accuracy)


if __name__ == '__main__':
    main(sys.argv[1:])
