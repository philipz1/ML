import pandas as pd
import numpy as np
import KNN

data = pd.read_csv("3Ddata.txt", sep=r"\s+", header = None)
n, p = data.shape
npdata = np.array([data.ix[:,i] for i in range(p-1)]).transpose()

def construct_mesh(data, graph):
	n, p = data.shape
	A_dist = np.zeros((n, n))

	for i, p1 in enumerate(data):
		for j, p2 in enumerate(data):
			if np.array_equal(p1, p2):
				A_dist[i,j] = 0
			elif tuple(p2) in graph[tuple(p1)]:
				A_dist[i,j] = 
				# A_dist[i,j] = graph[tuple(p1)][tuple(p2)]
			else:
				A_dist[i,j] = np.inf

	return A_dist

def le(data, k = 10):
	graph = KNN.knn(data, k)
