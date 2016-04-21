import pandas as pd
import numpy as np
import KNN

def do_gram(xs, k):
	'''
	computes the centered Gram Matrix
	G~ = -1/2 PDP
	'''
	n, p = xs.shape
	gram = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			gram[i,j] = np.dot(xs[i], xs[j])

	if k > p:
		gram = gram + 1e-3 * np.eye(k)

	return gram

def construct_knn_vector(x, graph, k):
	n = len(graph.keys())
	d = len(x)
	local = np.zeros((k,d), dtype = object)

	for i, key2 in enumerate(list(graph[x].keys())):
		local[i] = key2

	return local

def reconstruct(data, weights):
	'''
	Accepts the data and a dictionary of dictionaries of weights
	{p1: {p2: distance}} is the structure
	'''
	n, p = data.shape
	recon = np.zeros((n, n))

	for i, p1 in enumerate(data):
		for j, p2 in enumerate(data):
			if np.array_equal(p1, p2):
				recon[i,j] = 0
			elif tuple(p2) in weights[tuple(p1)]:
				recon[i,j] = weights[tuple(p1)][tuple(p2)]
			else:
				recon[i,j] = 0

	return recon

def lle(data, k = 10, target_dim = 2):
	p = data.shape[1]
	graph = KNN.knn(data, k)
 	
	n = len(graph.keys())
	weights_vec = np.zeros((n,k))
	weights_dict = {}
	locals_ = np.zeros((n,k))
	for i, key in enumerate(list(graph.keys())):
		local = construct_knn_vector(key, graph, k)
		local_centered = local - np.repeat(np.array(key).reshape([1, p]), k, axis = 0)
		gram = do_gram(local_centered, k)

		w_num = np.dot(np.linalg.inv(gram), np.ones(gram.shape[0]).T)
		w = w_num / w_num.sum()
		weights_vec[i] = w

		temp_dict = {}
		for q in range(len(local)):
			temp_dict[tuple(local[q])] = w[q]
		weights_dict[tuple(key)] = temp_dict

	weights = reconstruct(data, weights_dict)

	M = np.dot((np.identity(n) - weights).T, (np.identity(n) - weights))
	eigvals, eigvecs = np.linalg.eigh(M)

	index = np.argsort(eigvals)[::1]
	eigvals = eigvals[index]
	eigvecs = eigvecs[:,index]

	return eigvecs[:,1:target_dim + 1]