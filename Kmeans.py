__authors__ = ['1354223', '1571136', '1563587']
__group__ = 'DM.18'

import numpy as np
import math

from scipy.spatial.distance import cdist

import utils
import utils_data
import time


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictÂșionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self._init_centroids()

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """

        X[:] = X.astype(np.float64)

        self.og_shape = X.shape

        if len(X.shape) > 2:
            self.X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
        else:
            self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        self.centroids = np.zeros((self.K, self.X.shape[1]), dtype=np.float64)
        self.old_centroids = np.zeros((self.K, self.X.shape[1]), dtype=np.float64)

        if self.options['km_init'].lower() == 'first':
            for i in range(self.K):
                for x in self.X:
                    if x.tolist() not in self.centroids[0:i + 1].tolist():
                        self.centroids[i] = x
                        break

        if self.options['km_init'].lower() == 'random':
            repetits = True
            while repetits:
                self.centroids = np.random.rand(self.K, self.X.shape[1])

                repetits = False
                for i in range(self.K):
                    for j in range(i + 1, self.K):
                        if np.all(self.centroids[i] == self.centroids[j]):
                            repetits = True


        if self.options['km_init'].lower() == 'custom':
            minimum = np.min(self.X, axis=0)
            maximum = np.max(self.X, axis=0)
            line = maximum - minimum
            part = line / (self.K - 1)
            for i in range(self.K):
                self.centroids[i] = minimum + part * i

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """

        self.labels = np.random.randint(self.K, size=self.X.shape[0])

        distances = distance(self.X, self.centroids)

        self.labels[:] = np.argmin(distances, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """

        self.old_centroids[:] = self.centroids

        for c in range(self.K):
            point_idexes = np.where(self.labels == c)
            points_of_class = self.X[point_idexes]
            if points_of_class.size == 0:
                self.centroids[c] = 0
            else:
                self.centroids[c] = points_of_class.mean(axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """

        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """

        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges():
                break

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """

        wcd = 0
        for i, point in enumerate(self.X):
            c = self.labels[i]
            diff = (point - self.centroids[c])
            wcd += np.matmul(diff, diff.transpose())

        return wcd / len(self.X)

    def interClassDistance(self):
        icd = 0

        for c1 in range(self.K):
            for c2 in range(self.K):
                if c1 != c2:
                    point_idexes = np.where(self.labels == c1)
                    points_of_class1 = self.X[point_idexes]
                    point_idexes = np.where(self.labels == c2)
                    points_of_class2 = self.X[point_idexes]

                    distance = cdist(points_of_class1, points_of_class2, metric='euclidean')
                    icd += distance.sum()

        return icd / 2


    def fisherDiscriminant(self):
        wcd = self.whitinClassDistance()
        icd = self.interClassDistance()
        return wcd / icd


    def calculateHeuristic(self, heuristic):
        if heuristic == 'within-class-distance':
            return self.whitinClassDistance()
        elif heuristic == 'inter-class-distance':
            return self.interClassDistance()
        elif heuristic == "fisher":
            return self.fisherDiscriminant()


    def shouldStop(self, old_h, h, heuristic, threshold):
        if heuristic == 'within-class-distance':
            dec = 100 * (h / old_h)
            if 100 - dec < threshold:
                return True
        elif heuristic == 'inter-class-distance':
            inc = 100 * (h / old_h)
            if inc < threshold:
                return True
        return False

    def find_bestK(self, max_K, heuristic='within-class-distance', threshold=20):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """

        old_h = None
        for k in range(2, max_K + 1):
            self.K = k
            self._init_centroids()
            self.num_iter = 0
            self.fit()
            h = self.calculateHeuristic(heuristic)
            if old_h != None:
                if self.shouldStop(old_h, h, heuristic, threshold):
                    # la diferencia ya no es significativa y nos quedamos con el anterior
                    self.K = k - 1
                    break
            old_h = h

            # utils_data.visualize_k_means(self, self.og_shape)


def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    arr = np.zeros((X.shape[0], C.shape[0]))

    for i in range(len(C)):
        arr[:, i] = np.linalg.norm(X - C[i], axis=1)

    return arr


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    probabilities = utils.get_color_prob(centroids)
    colors = []
    for p in probabilities:
        max_index = np.argmax(p)
        colors.append(utils.colors[max_index])

    return colors
