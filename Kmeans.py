__authors__ = ['1354223', '1571136', '1563587']
__group__ = 'DM.18'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self._init_centroids()

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################






    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """


        X[:] = X.astype(np.float64)

        # TODO: fer el que ha dit l'Ali de files*columnes, channel

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
            options['fitting'] = 'WCD'  #within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################




    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if self.options['km_init'].lower() == 'first':
            self.centroids = np.zeros((self.K, self.X.shape[1]), dtype=np.float64)
            self.old_centroids = np.zeros((self.K, self.X.shape[1]), dtype=np.float64)

            for i in range(self.K):
                for x in self.X:
                    if x.tolist() not in self.centroids[0:i + 1].tolist():
                        self.centroids[i] = x
                        self.old_centroids[i] = x
                        break

        else:
            repetits = True
            while repetits:
                self.centroids = np.random.rand(self.K, self.X.shape[1])

                repetits = False
                for i in range(self.K):
                    for j in range(i + 1, self.K):
                        if self.centroids[i] == self.centroids[j]:
                            repetits = True

            self.old_centroids = self.centroids


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.labels = np.random.randint(self.K, size=self.X.shape[0])

        distances = distance(self.X, self.centroids)

        self.labels[:] = np.argmin(distances, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.old_centroids[:] = self.centroids

        for c in range(self.K):
            point_idexes = np.where(self.labels == c)
            points_of_class = self.X[point_idexes]
            self.centroids[c] = points_of_class.mean(axis=0)


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return (self.centroids == self.old_centroids).all()

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        while True:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges():
                break


    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """


        '''
        wcd = 0
        
        for c in range(self.K):
            points_of_class = self.X[np.where(self.labels == c)]
            wcd += np.matmul(
                        np.reshape(points_of_class[None, :] - points_of_class[:, None], (points_of_class.shape[0]**2, points_of_class.shape[1])),
                        np.transpose(np.reshape(points_of_class[None, :] - points_of_class[:, None], (points_of_class.shape[0]**2, points_of_class.shape[1])))
                    ).sum()
        '''
        '''
        avg = []
        for c in range(self.K):
            points_of_class = self.X[np.where(self.labels == c)]
            sum = 0
            for i, pointa in enumerate(points_of_class):
                for j, pointb in enumerate(points_of_class):
                    sum += (pointa - pointb).sum()
            avg.append(sum/len(points_of_class))
        wcd = sum(avg)/len(avg)
        '''
        '''
        print("aaaaaaaaaaa")
        wcd = 0
        for c in range(self.K):
            points_of_class = self.X[np.where(self.labels == c)]
            for point in points_of_class:
                diff = (- points_of_class + point)
                wcd += np.matmul(diff, diff.transpose()).sum()
        '''
        wcd = 0
        for i, point in enumerate(self.X):
            c = self.labels[i]
            diff = (point - self.centroids[c])
            wcd += np.matmul(diff, diff.transpose())

        return wcd / len(self.X)

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        old_wcd = None
        for k in range(2, max_K + 1):
            self.K = k
            self._init_centroids()
            self.fit()
            wcd = self.whitinClassDistance()
            if old_wcd != None:
                dec = 100 * (wcd / old_wcd)
                if 100 - dec < 20:
                    # la diferencia ya no es significativa y nos quedamos con el anterior
                    self.K = k - 1
                    break
            old_wcd = wcd


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

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    arr = np.zeros((X.shape[0], C.shape[0]))
    for i, point in enumerate(X):
        for j, centroid in enumerate(C):
            arr[i, j] = np.linalg.norm(point - centroid)

    return arr

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    probabilities = utils.get_color_prob(centroids)
    colors = []
    for p in probabilities:
        max_index = np.argmax(p)
        colors.append(utils.colors[max_index])

    return colors

