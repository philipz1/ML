'''
This fucking algorithm / how i learned numpy
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import multivariate_normal
import numpy as np
import time
import random
import kMeans

def random_points(data, k):
    '''
    Chooses k random points from a set
    '''
    n, p = data.shape
    points = np.random.choice(np.arange(0, len(data)), k) 
    return data[np.ix_(list(points), list(np.arange(0, p)))]

def kmix(xs, k, tolerance = 0.01, max_iter=100, verbose = True, init = 'random'):
    n, p = xs.shape #n total data points, p is dimension

    if init == 'kmeans':
        if verbose == True:
            print('Initializing points with K-Means++.')
        mus = list(kMeans.kmeans(xs, k, plus = True, verbose = verbose).keys())
    else:
        if verbose == True:
            print('Initializing points randomly.')
        mus = random_points(xs, k)

    sigmas = [np.array([[1, 0],[0, 1]])] * k
    pis = [1 / k] * k

    ll_old = 0
    for i in range(max_iter):
        if verbose == True:
            print('Iteration {} | loglikelihood {}'.format(i, ll_old))

        '''
        pij is a kxn array. We iterate over the clusters and then over each data
        point. pij /= pij.sum(0) divides the columns by the sum of the columns
        '''
        pij = np.zeros((k, n))
        for j in range(len(mus)):
            for i in range(n):
                pij[j, i] = pis[j] * multivariate_normal.pdf(xs[i], mus[j], sigmas[j])
        pij /= pij.sum(0)

        '''
        we iterate over the clusters and then over the data again, and pis is
        initially the sum of all the data points per cluster.
        Then we divide each cluster by the length of the total data
        '''
        pis = np.zeros(k)
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += pij[j, i]
        pis /= n

        '''
        mus is initially the points weighted by the pij.
        Then we divide mus by the sum of the weights in the cluster
        '''
        mus = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mus[j] += pij[j, i] * xs[i]
            mus[j] /= pij[j, :].sum()

        '''
        sigmas is initially k p dimensional square matrices
        Then it becomes the sum of the weighted covariances
        Then we divide by the sum of the weights in the cluster
        '''
        sigmas = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(xs[i]- mus[j], (2,1))
                sigmas[j] += pij[j, i] * np.dot(ys, ys.T)
            sigmas[j] /= pij[j,:].sum()

        '''
        iterate over all data points then over all clusters
        add the weighted things per cluster, then log
        '''
        ll_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pis[j] * multivariate_normal.pdf(xs[i], mus[j], sigmas[j])
            ll_new += np.log(s)
        
        if np.abs(ll_new - ll_old) <= tolerance:
            break
        ll_old = ll_new

    return_dict = {}
    for i in range(n):
        max_tracker = []
        for j in range(k):
            max_tracker.append(pis[j] * multivariate_normal(mus[j], sigmas[j]).pdf(xs[i]))
        index = max_tracker.index(max(max_tracker))
        if not (tuple(mus[index])) in return_dict.keys():
            return_dict[tuple(mus[index])] = [xs[i]]
        else:
            return_dict[tuple(mus[index])].append(xs[i])

    return return_dict