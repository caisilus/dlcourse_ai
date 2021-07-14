import numpy as np
import queue
import heapq
from scipy import stats

class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)


    def l1_distance(self, v, u):
        '''
        Computes l1 distance between two vectors
        '''
        l1 = 0
        for i in range(v.shape[0]):
            l1 += abs(v[i] - u[i])
        return l1


    def l1_dist_matr(self, v, m):
        u = np.sum(np.abs(m - v), axis=1)
        #print(u)
        #print(u.shape)
        return u


    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # Fill dists[i_test][i_train]
                dists[i_test][i_train] = self.l1_distance(self.train_X[i_train], X[i_test])
        return dists


    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            #vv = np.tile(X[i_test, :], (num_train, 1))
            #print(vv.shape)
            dists[i_test, :] = np.sum(np.abs(self.train_X - X[i_test, :]), axis=1)
        return dists
        

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        #Implement computing all distances with no loops!
        dists = np.sum(np.abs(X[:,None] - self.train_X), axis=-1)
        #dist_fun = np.vectorize(self.l1_dist_matr) #, signature='(n),(m)->(m)')
        #dists = dist_fun(X, self.train_X)
        
        return dists


    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        def get_second(t):
            return t[1]

        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # Implement choosing best class based on k
            # nearest training samples
            num_train = self.train_X.shape[0]
            q = [(dists[i][j], self.train_y[j]) for j in range(num_train)]
            arr = np.array(heapq.nsmallest(self.k, q, (lambda x: x[0])))
            k_smallest_fun = np.vectorize(get_second, signature='(n)->()')
            k_smallest = k_smallest_fun(arr)
            mode = (stats.mode(k_smallest))[0] #most recent element
            #print(bool(mode))
            pred[i] = bool(mode)

        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''

        def get_second(t):
            return t[1]

        num_test = dists.shape[0]
        num_train = dists.shape[1]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # Implement choosing best class based on k
            # nearest training samples
            q = [(dists[i][j], self.train_y[j]) for j in range(num_train)]
            arr = np.array(heapq.nsmallest(self.k, q, (lambda x: x[0])))
            k_smallest_fun = np.vectorize(get_second, signature='(n)->()')
            k_smallest = k_smallest_fun(arr)
            mode = (stats.mode(k_smallest))[0]
            pred[i] = int(mode)
            #print(pred[i])
        return pred
