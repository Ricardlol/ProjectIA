__authors__ = ['1354223', '1571136', '1563587']
__group__ = 'DM.18'

import copy

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt
import cv2
import time

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

def Kmean_statistics(kmeans, Kmax, heuristic):
    wcds = []
    icds = []
    fishers = []
    iters = []

    for k in range(2, Kmax + 1):
        kmeans.K = k
        kmeans._init_centroids()
        kmeans.num_iter = 0
        kmeans.fit()
        if heuristic == "wcd":
            wcd = kmeans.whitinClassDistance()
            wcds.append(wcd)

        if heuristic == "inter":
            icd = kmeans.interClassDistance()
            icds.append(icd)

        if heuristic == "fisher":
            fisher = kmeans.fisherDiscriminant()
            fishers.append(fisher)

        iters.append(kmeans.num_iter)



    Ks = range(2, Kmax + 1)

    if heuristic == "wcd":
        return iters, wcds

    if heuristic == "inter":
        return iters, icds

    if heuristic == "fisher":
        return iters, fishers

    return iters, wcds



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

def printPlot2DAverage(x, y1, y2, y1label, y2label, km_init):
    plt.title("KMeans : " + km_init)
    plt.xlabel("K")
    plt.ylabel(y1label)
    plt.plot(x, y1)
    plt.show()

    plt.title("KMeans : " + km_init)
    plt.xlabel("K")
    plt.ylabel(y2label)
    plt.plot(x, y2)
    plt.show()

    plt.title("Kmeans")
    plt.xlabel("K")
    plt.ylabel("Average")
    plt.plot(x, y1, label=y1label)
    plt.plot( x,y2, label=y2label)
    plt.legend()
    plt.show()

def printPlot2DMultipleImages(x, yN, title):
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Average")
    for yIndex, y in enumerate(yN):
        plt.plot(x, y, label="image: " + str(yIndex))
    plt.legend()
    plt.show()

def printPlot2DMultipleKMinit(x, yN, title, legendList):
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Average Iterations")
    for yIndex, y in enumerate(yN):
        plt.plot(x, y, label=legendList[yIndex])
    plt.legend()
    plt.show()

def allPlotsKMeansStatistics(test_imgs, km_init, heuristic):
    imgNum = 5
    maxK = 10
    totalIterations = [0] * maxK
    totalDistance = [0] * maxK
    totalKs = [x for x in range(2, maxK + 1)]
    iterationsList = []
    distanceList = []
    for x in range(imgNum):
        totalIterationsAux = copy.deepcopy(totalIterations)
        totalDistanceAux = copy.deepcopy(totalDistance)
        kmeans = Kmeans.KMeans(test_imgs[x], options={'km_init': km_init})
        newIterations, newDistance = Kmean_statistics(kmeans, maxK, heuristic)
        iterationsList.append(newIterations)
        totalIterations = [x + y for x, y in zip(totalIterationsAux, newIterations)]
        distanceList.append(newDistance)
        totalDistance = [x + y for x, y in zip(totalDistanceAux, newDistance)]

    averageIterations = [x / imgNum for x in totalIterations]
    averageDistance = [x / imgNum for x in totalDistance]

    printPlot2DAverage(totalKs, averageIterations, averageDistance, "Iterations", "Distance", km_init)

    printPlot2DMultipleImages(totalKs, iterationsList, "Iterations by image : " + km_init)
    printPlot2DMultipleImages(totalKs, distanceList, "Distance by image : " + km_init)

    return averageIterations, averageDistance


if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/',
                                                                   gt_json='./images/gt.json')

    # List with all the existant classes
    #classes = list(set(list(train_class_labels) + list(test_class_labels)))

    #found = Retrieval_by_color(test_imgs, test_color_labels, ['Green', 'Blue'])

    #visualize_retrieval(found, len(found))

    #found = Retrieval_by_shape(test_imgs, test_class_labels, ['Shorts'])

    #visualize_retrieval(found, len(found))

    #found = Retrieval_combined(test_imgs, test_class_labels, test_color_labels, ['Shorts'], ['Green'])

    #visualize_retrieval(found, len(found))

    maxK = 10
    totalKs = [x for x in range(2, maxK + 1)]

    kminitIteration = []
    kminitDistance = []
    kminitOptions = ["first", "custom", "random"]
    for kminit in kminitOptions:
        iterations, distance = allPlotsKMeansStatistics(test_imgs, kminit, "wcd")
        kminitDistance.append(distance)
        kminitIteration.append(iterations)
    printPlot2DMultipleKMinit(totalKs, kminitIteration, "Iterations by km_init (heuristics = WCD)", kminitOptions)
    printPlot2DMultipleKMinit(totalKs, kminitDistance, "Iterations by km_init : WCD", kminitOptions)

    kminitIteration = []
    kminitDistance = []
    kminitOptions = ["first", "custom", "random"]
    for kminit in kminitOptions:
        iterations, distance = allPlotsKMeansStatistics(test_imgs, kminit, "inter")
        kminitDistance.append(distance)
        kminitIteration.append(iterations)
    printPlot2DMultipleKMinit(totalKs, kminitIteration, "Iterations by km_init (heuristics = inter)", kminitOptions)
    printPlot2DMultipleKMinit(totalKs, kminitDistance, "Iterations by km_init : inter", kminitOptions)

    kminitIteration = []
    kminitDistance = []
    kminitOptions = ["first", "custom", "random"]
    for kminit in kminitOptions:
        iterations, distance = allPlotsKMeansStatistics(test_imgs, kminit, "fisher")
        kminitDistance.append(distance)
        kminitIteration.append(iterations)
    printPlot2DMultipleKMinit(totalKs, kminitIteration, "Iterations by km_init (heuristics = fisher)", kminitOptions)
    printPlot2DMultipleKMinit(totalKs, kminitDistance, "Iterations by km_init : fisher", kminitOptions)


    # kminitIteration = []
    # kminitDistance = []
    # heuristics = ["wcd", "inter", "fisher"]
    # for heur in heuristics:
    #     iterations, distance = allPlotsKMeansStatistics(test_imgs, "custom", heur)
    #     kminitDistance.append(distance)
    #     kminitIteration.append(iterations)
    #printPlot2DMultipleKMinit(totalKs, kminitIteration, "Iterations by heuristics (km_init = custom)", heuristics)
    #printPlot2DMultipleKMinit(totalKs, kminitDistance, "Iterations by heuristics : (km_init = custom)", heuristics)


    # Should be 100%
    #rate, incorrect_indexes = Get_shape_accuracy(test_class_labels, test_class_labels)
    #print("Shape accuracy:", rate)
    #print("Incorrect predictions:", incorrect_indexes)
    #rate, incorrect_indexes = Get_color_accuracy(test_color_labels, test_color_labels)
    #print("Color accuracy:", rate)
    #print("Incorrect predictions:", incorrect_indexes)


    # parte Find_BestK
'''
    img = test_imgs[1]
    kmeans = Kmeans.KMeans(img)
    kmeans.find_bestK(100, heuristic='fisher', threshold=5)
    kmeans.fit()
    ax = Plot3DCloud(kmeans)
    plt.show()
    visualize_k_means(kmeans, img.shape)
'''
