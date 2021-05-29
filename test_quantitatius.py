__authors__ = ['1354223', '1571136', '1563587']
__group__ = 'DM.18'

import numpy as np
from  Kmeans import KMeans, get_colors
from KNN import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt
from time import perf_counter

import cv2

from time import perf_counter

from my_labeling import Get_shape_accuracy, Get_color_accuracy

if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
        test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/',
                                                                   gt_json='./images/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    print(len(test_imgs))

    imgNum = 150
    images_to_test = test_imgs[0:imgNum]
    expected_class_labels = test_class_labels[0:imgNum]
    expected_color_labels = test_color_labels[0:imgNum]

    accuracyList = []
    thresholdList = []
    timeList = []

    for thr in range(5, 50, 5):

        # SETUP KMEANS
        start = perf_counter()
        imgs = []
        labels = []
        for i, img in enumerate(images_to_test):
            kmeans = KMeans(img, 0)
            kmeans.find_bestK(len(train_color_labels), threshold=thr)
            k = kmeans.K
            print("K (expected, actual) = ({}, {})".format(len(expected_color_labels[i]), k))
            kmeans = KMeans(img, k)
            kmeans.fit()
            imgs.append(img)
            labels.append(get_colors(kmeans.centroids))

        stop = perf_counter()
        time = stop - start
        # npimgs = np.array(imgs)
        nplabels = np.array(labels, dtype=object)

        # # SETUP KNN
        # knn = KNN(train_imgs, train_class_labels)
        # class_labels = knn.predict(images_to_test, 5)

        # print("Shape accuracy:", Get_shape_accuracy(class_labels, expected_class_labels))
        accuracyList.append(Get_color_accuracy(nplabels, expected_color_labels)[0])
        thresholdList.append(thr)
        timeList.append(time)
        print("Color accuracy:", Get_color_accuracy(nplabels, expected_color_labels))

    plt.title("KMeans")

    plt.xlabel("Threshold %")
    plt.ylabel("Accuracy %")

    plt.plot(thresholdList, accuracyList)
    plt.show()

    plt.plot(thresholdList, timeList)
    plt.title("KMeans")

    plt.xlabel("Threshold %")
    plt.ylabel("Time in seconds")

    plt.show()

