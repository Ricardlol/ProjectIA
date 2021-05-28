__authors__ = ['1354223', '1571136', '1563587']
__group__ = 'DM.18'

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt
import cv2


## You can start coding your functions here


def Retrieval_by_color(imatges, etiquetes, colors):
    matches = []
    for i, labels in enumerate(etiquetes):
        if all(c in labels for c in colors):
            matches.append(imatges[i])

    return matches


def Retrieval_by_shape(imatges, etiquetes, classes):
    matches = []
    for i, labels in enumerate(etiquetes):
        if all(c in labels for c in classes):
            matches.append(imatges[i])

    return matches

def Retrieval_combined(imatges, class_labels, color_labels, classes, colors):
    matches = []
    for i, image in enumerate(imatges):
        if all(c in class_labels[i] for c in classes) and all(c in color_labels[i] for c in colors):
            matches.append(image)

    return matches


def Kmean_statistics(kmeans, Kmax):
    wcds = []
    icds = []
    iters = []
    for k in range(2, Kmax + 1):
        kmeans.K = k
        kmeans._init_centroids()
        kmeans.num_iter = 0
        kmeans.fit()
        icd = kmeans.interClassDistance()
        wcd = kmeans.whitinClassDistance()
        icds.append(icd)
        wcds.append(wcd)
        iters.append(kmeans.num_iter)

    Ks = range(2, Kmax + 1)

    _, ax = plt.subplots()
    ax.plot(Ks, iters)
    plt.title("Init centroids: " + kmeans.options['km_init'])
    plt.xlabel("K")
    plt.ylabel("Iterations")
    plt.legend()
    plt.show()

    _, ax = plt.subplots()
    ax.plot(Ks, wcds)
    plt.title("Init centroids: " + kmeans.options['km_init'])
    plt.xlabel("K")
    plt.ylabel("Within class distance")
    plt.legend()
    plt.show()

    _, ax = plt.subplots()
    ax.plot(Ks, icds)
    plt.title("Init centroids: " + kmeans.options['km_init'])
    plt.xlabel("K")
    plt.ylabel("Inter class distance")
    plt.legend()
    plt.show()

    _, ax = plt.subplots()
    ax.plot(Ks, icds, wcds)
    plt.title("Init centroids: " + kmeans.options['km_init'])
    plt.xlabel("K")
    plt.ylabel("Inter class distance")
    plt.legend()
    plt.show()

def Get_shape_accuracy(actual_class_labels, expected_class_labels):
    corrects = 0
    incorrect_indexes = []
    for i in range(len(actual_class_labels)):
        if actual_class_labels[i] == expected_class_labels[i]:
            corrects += 1
        else:
            incorrect_indexes.append(i)

    return corrects / len(actual_class_labels), incorrect_indexes

def Get_color_accuracy(actual_color_labels, expected_color_labels):
    corrects = 0
    incorrect_indexes = []
    for i in range(len(actual_color_labels)):
        if all(color in set(actual_color_labels[i]) for color in set(expected_color_labels[i])):
            corrects += 1
        else:
            incorrect_indexes.append(i)

    return corrects / len(actual_color_labels), incorrect_indexes



if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/',
                                                                   gt_json='./images/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    found = Retrieval_by_color(test_imgs, test_color_labels, ['Green', 'Blue'])

    visualize_retrieval(found, len(found))

    found = Retrieval_by_shape(test_imgs, test_class_labels, ['Shorts'])

    visualize_retrieval(found, len(found))

    found = Retrieval_combined(test_imgs, test_class_labels, test_color_labels, ['Shorts'], ['Green'])

    visualize_retrieval(found, len(found))


    kmeans = Kmeans.KMeans(test_imgs[3], options={'km_init': 'first'})
    Kmean_statistics(kmeans, 10)

    # Should be 100%
    rate, incorrect_indexes = Get_shape_accuracy(test_class_labels, test_class_labels)
    print("Shape accuracy:", rate)
    print("Incorrect predictions:", incorrect_indexes)
    rate, incorrect_indexes = Get_color_accuracy(test_color_labels, test_color_labels)
    print("Color accuracy:", rate)
    print("Incorrect predictions:", incorrect_indexes)

'''
    # parte Find_BestK
    img = test_imgs[1]
    kmeans = Kmeans.KMeans(img)
    kmeans.find_bestK(100, heuristic='icd', threshold=5)
    kmeans.fit()
    ax = Plot3DCloud(kmeans)
    plt.show()
    visualize_k_means(kmeans, img.shape)
'''
