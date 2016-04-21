import pandas as pd
import numpy as np
import time
import KNN

def dist(p1, p2):
	'''
	uses the distance squared metric
	'''
	J = 0
	for i in range(len(p1)):
		J += (p1[i] - p2[i]) ** 2
	return np.sqrt(J)

def construct_A(data, graph):
	n, p = data.shape
	A = np.zeros((n, n), dtype = object)
	A_dist = np.zeros((n, n))

	for i, p1 in enumerate(data):
		for j, p2 in enumerate(data):
			if np.array_equal(p1, p2):
				A_dist[i,j] = 0
			elif tuple(p2) in graph[tuple(p1)]:
				A_dist[i,j] = graph[tuple(p1)][tuple(p2)]
			else:
				A_dist[i,j] = np.inf

	return A_dist

def shortest_dist_weight(A):
	#Floyd-Warshall Algorithm
	n, p = A.shape
	A_short = np.zeros((n, n))
	for k in range(n):
		for i in range(n):
			for j in range(n):
				if A[i,j] > A[i,k] + A[k,j]:
					A[i,j] = A[i,k] + A[k,j]

	return A

# def do_gram(xs):
# 	'''
# 	computes the centered Gram Matrix
# 	G~ = -1/2 PDP
# 	'''
# 	n, p = xs.shape
# 	gram = np.zeros((n,n))
# 	for i in range(n):
# 		for j in range(n):
# 			gram[i,j] = np.dot(xs[i], xs[j])

# 	ones = np.ones(n).reshape([n,1])
# 	P = np.identity(n) - 1/n * np.dot(ones, ones.T)
# 	PG = np.dot(P, gram)
# 	PGP = np.dot(PG,P)
# 	return PGP

def do_gram_tilda(dists):
	'''
	computes the centered Gram Matrix
	G~ = -1/2 PDP
	'''
	n, p = dists.shape
	gram = np.zeros((n,n))
	ones = np.ones(n).reshape([n,1])
	P = np.identity(n) - 1/n * np.dot(ones, ones.T)
	PD = np.dot(P, dists**2)
	PDP = -1/2 * np.dot(PD, P)

	return PDP

def isomap(data, k = 10, target_dim = 2, load = False, save = False):
	if load == False:
		graph = KNN.knn(data, k)
		A = construct_A(data, graph)
		dists = shortest_dist_weight(A)
		if save != False:
			np.save(save, dists)
	else:
		dists = np.load(load)

	gram_tilda = do_gram_tilda(dists)
	eigvals, eigvecs = np.linalg.eigh(gram_tilda)

	index = np.argsort(eigvals)[::-1]
	eigvals = eigvals[index]
	eigvecs = eigvecs[:,index]

	return eigvecs[:,0:target_dim]

