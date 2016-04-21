import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import random
matplotlib.style.use('ggplot')
pd.set_option('display.width', 175)

def dist(p1, p2):
	'''
	uses the distance squared metric
	'''
	J = 0
	for i in range(len(p1)):
		J += (p1[i] - p2[i]) ** 2
	return J

def random_points(xs, k):
    '''
    Chooses k random points from a set
    '''
    n, p = xs.shape
    points = np.random.choice(np.arange(0, len(xs)), k) 
    return xs[np.ix_(list(points), list(np.arange(0, p)))]

def kmeanplusplus(xs, k):
	n, p = xs.shape
	points = []
	points.append(np.random.choice(np.arange(0, len(xs)), 1))

	while len(points) != k:
		numerators = []
		for x in xs:
			min_tracker = []
			for point in points:
				min_tracker.append([dist(x, xs[np.ix_(point, list(np.arange(0, p)))][0]), point])
			argmin = min(min_tracker)
			numerators.append(argmin[0] ** 2)
		numerators = np.array(numerators)
		denominator = numerators.sum()
		points.append(np.random.choice(np.arange(0, len(xs)), 1, p=numerators/denominator))
	return xs[np.ix_(np.array(points).T[0], list(np.arange(0, p)))]

def kmeans(xs, k, distortion = False, plus = False, max_iter = 100, verbose = True):
	if verbose == True:
		if plus == True:
			print('K-Means++ Algorithm')
		else:
			print('K-Means Algorithm')

	distortion_sum = []
	#generating starting points
	if plus == True:
		mus = kmeanplusplus(xs, k)
	else:
		mus = random_points(xs, k)

	for i in range(max_iter):
		points_dict = {}
		for center in mus:
			points_dict[tuple(center)] = []

		#argmin
		for x in xs:
			min_tracker = []
			for center in mus:
				min_tracker.append([dist(x, center), center])
			argmin = min(min_tracker)
			points_dict[tuple(argmin[1])].append(np.array(x))

		#new centers
		previous_mus = mus
		mus = []
		for p in previous_mus:
			cardinality = len(points_dict[tuple(p)])
			temp = np.array(points_dict[tuple(p)])
			new_point = (temp.sum(0) / cardinality)
			mus.append(new_point)
		mus = np.array(mus)

		# if distortion == True:
		dd = 0
		for key in points_dict:
			for value in points_dict[key]:
				dd += dist(key, value)
		distortion_sum.append(dd)
		
		if verbose == True:
			print("Iteration {} | Distortion {}".format(i, dd))

		if np.array_equal(previous_mus, mus):
			break

	if distortion == True:
		return points_dict, distortion_sum
	else:
		return points_dict