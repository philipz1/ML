import pandas as pd
import numpy as np
import KNN
import scipy

# data = pd.read_csv("3Ddata.txt", sep=r"\s+", header = None)
# n, p = data.shape
# npdata = np.array([data.ix[:,i] for i in range(p-1)]).transpose()

def dist(p1, p2):
	'''
	uses the distance squared metric
	'''
	J = 0
	for i in range(len(p1)):
		J += (p1[i] - p2[i]) ** 2
	return np.e(-J/4)

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

def construct_mesh(data, graph):
	n, p = data.shape
	A_dist = np.zeros((n, n))

	for i, p1 in enumerate(data):
		for j, p2 in enumerate(data):
			if np.array_equal(p1, p2):
				A_dist[i,j] = 0
			elif tuple(p2) in graph[tuple(p1)]:
				A_dist[i,j] = graph[tuple(p1)][tuple(p2)]
			else:
				A_dist[i,j] = 0

	return A_dist

def construct_degree(A):
	n, p = A.shape
	D = np.zeros((n,n))
	for i in range(n):
		D[i,i] = np.array([A[i,j] for j in range(n)]).sum()

	return D

def le(data, k = 10, target_dim = 2):
	graph = KNN.knn(data, k)
	A = construct_mesh(data, graph)
	from sklearn import manifold
	return(manifold.spectral_embedding(A, 2))

	D = construct_degree(A)
	L = D - A

	eigvals, eigvecs = scipy.linalg.eigh(A, L)

	index = np.argsort(eigvals)[::-1]
	eigvals = eigvals[index]
	eigvecs = eigvecs[:,index]
	
	return eigvecs[:,1:target_dim + 1]

# print(le(npdata))