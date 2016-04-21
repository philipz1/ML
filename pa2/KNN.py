import pandas as pd
import numpy as np
import time

def dist(p1, p2):
	'''
	uses the distance squared metric
	'''
	J = 0
	for i in range(len(p1)):
		J += (p1[i] - p2[i]) ** 2
	return np.sqrt(J)

def knn(data, k):
	'''
	returns a dictionary of dictionaries where key: key: distance between keys 
	'''
	n, p = data.shape
	knn_dict = {}

	for i, p1 in enumerate(data):
		k_list = []
		k_dict = {}
		for j, p2 in enumerate(data):
			k_list.append([p2, dist(p1,p2)])

		k_list.sort(key = lambda x : x[1])

		for i in range(1, k+1): #gets rid of self
			k_dict[tuple(k_list[i][0])] = k_list[i][1]
		knn_dict[tuple(p1)] = k_dict

	return knn_dict


# def knn(data, k):
# 	'''
# 	returns a dictionary of dictionaries where key: key: distance between keys 
# 	'''
# 	n, p = data.shape
# 	knn_dict = {}

# 	for i, p1 in enumerate(data):
# 		k_list = [0] * k
# 		k_dist = [0] * k
# 		k_dict = {}
# 		for j, p2 in enumerate(data):
# 			zero_check = sum([1 if type(i).__module__ == 'numpy' else 0 for i in k_list]) != k

# 			if zero_check and dist(p1, p2) != 0:
# 				k_list[k_dist.index(0)] = p2
# 				k_dist[k_dist.index(0)] = dist(p1, p2)

# 			elif dist(p1, p2) < max(k_dist) and dist(p1, p2) != 0:
# 				k_list[k_dist.index(max(k_dist))] = p2
# 				k_dist[k_dist.index(max(k_dist))] = dist(p1, p2)

# 		for i in range(len(k_list)):
# 			k_dict[tuple(k_list[i])] = k_dist[i]
# 		knn_dict[tuple(p1)] = k_dict

# 	return knn_dict