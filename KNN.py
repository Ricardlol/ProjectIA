__authors__ = ['1354223', '1571136', '1563587']
__group__ = 'DM.18'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

from scipy.ndimage.filters import gaussian_filter, sobel

from matplotlib.colors import rgb_to_hsv


class KNN:
    def __init__(self, train_data, labels, image_transformation=None):

        self.image_transformation = image_transformation
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        train_data[:] = train_data.astype(np.float64)

        train_data = self.convolute_many(train_data, name=self.image_transformation)

        self.train_data = np.reshape(train_data, (
        train_data.shape[0], prod(train_data.shape[1:])))

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        test_data[:] = test_data.astype(np.float64)

        test_data = np.reshape(test_data,
                               (test_data.shape[0], prod(test_data.shape[1:])))

        distance = cdist(test_data, self.train_data, metric='euclidean')

        neighbors = np.argsort(distance, axis=1)

        self.neighbors = self.labels[neighbors[:, 0:k]]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """

        # unique, counts = np.unique(self.neighbors, return_counts=True, axis=0)

        classes = []

        for n in self.neighbors:
            unique, counts = np.unique(n, return_counts=True)
            classes.append(unique[np.argmax(counts)])

        return np.array(classes)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """

        test_data = self.convolute_many(test_data, name=self.image_transformation)

        self.get_k_neighbours(test_data, k)

        return self.get_class()

    def convolute_many(self, images, name=None):
        ogs = images.shape
        if name == 'hsv':
            new_images = np.zeros((ogs[0], ogs[1], ogs[2], ogs[3]))
            for i, img in enumerate(images):
                new_images[i] = self.to_hsv(images[i])
        if name == 'gauss':
            new_images = np.zeros((ogs[0], ogs[1], ogs[2], ogs[3]))
            for i, img in enumerate(images):
                new_images[i] = self.gauss(images[i])
        elif name == 'grey':
            new_images = np.zeros((ogs[0], ogs[1], ogs[2]))
            for i, img in enumerate(images):
                new_images[i] = rgb2gray(images[i])
        elif name == 'sobel':
            new_images = np.zeros((ogs[0], ogs[1], ogs[2], ogs[3] + 1))
            for i, img in enumerate(images):
                new_images[i] = self.sobel(images[i])
        else:
            new_images = images

        return new_images

    def gauss(self, data):
        data[:, :, 0] = gaussian_filter(data[:, :, 0], 3)
        data[:, :, 1] = gaussian_filter(data[:, :, 1], 3)
        data[:, :, 2] = gaussian_filter(data[:, :, 2], 3)
        return data

    def sobel(self, data):
        grey_image = rgb2gray(data)
        grey_sobel_h = sobel(grey_image, 0)
        grey_sobel_v = sobel(grey_image, 1)
        grey_sobel = np.hypot(grey_sobel_h, grey_sobel_v)
        grey_sobel *= 255.0 / np.max(grey_sobel)

        ogs = data.shape
        new_data = np.zeros((ogs[0], ogs[1], ogs[2] + 1))

        new_data[:, :, 0:3] = data
        new_data[:, :, 3] = grey_sobel

        return new_data

    def to_hsv(self, data):
        return rgb_to_hsv(data)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def prod(numbers):
    res = 1
    for n in numbers:
        res *= n
    return res
