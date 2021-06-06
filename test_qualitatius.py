__authors__ = ['1354223', '1571136', '1563587']
__group__ = 'DM.18'

import numpy as np
from  Kmeans import KMeans, get_colors
from KNN import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud, visualize_sobel
import matplotlib.pyplot as plt
import cv2

from my_labeling import Retrieval_by_color, Retrieval_by_shape, Retrieval_combined

if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
        test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/',
                                                                   gt_json='./images/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    print(len(test_imgs))

    images_to_test = test_imgs

    # SETUP KMEANS
    imgs = []
    labels = []
    for img in images_to_test:
        kmeans = KMeans(img, 0)
        kmeans.find_bestK(len(train_color_labels))
        k = kmeans.K
        kmeans = KMeans(img, k)
        kmeans.fit()
        imgs.append(img)
        labels.append(get_colors(kmeans.centroids))

    npimgs = np.array(imgs)
    nplabels = np.array(labels, dtype=object)

    # SETUP KNN
    knn = KNN(train_imgs, train_class_labels)
    class_labels, tested_images = knn.predict(images_to_test, 5)

    viewLen=12
    # RETRIEVE BY COLOR
    found = Retrieval_by_color(npimgs, nplabels, ['Green'])
    visualize_retrieval(found, min(len(found),viewLen ))

    # RETRIEVAL BY SHAPE
    found = Retrieval_by_shape(npimgs, class_labels, ['Shorts'])
    visualize_retrieval(found, min(len(found), viewLen))
    found = Retrieval_by_shape(npimgs, class_labels, ['Shorts'])
    visualize_sobel(found, min(len(found), viewLen))

    # RETRIEVAL COMBINED
    found = Retrieval_combined(npimgs, class_labels, nplabels, ['Shorts'], ['Green'])
    if len(found) < viewLen:
        viewLen = len(found)
    visualize_retrieval(found, viewLen)

