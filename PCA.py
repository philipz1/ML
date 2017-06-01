'''
Note thise assumes that the data will be in the form of numpy arrays
where the columns are the dimensions
'''
import pandas as pd
import numpy as np
import time

def center(data):
	'''
	centers the data
	'''
	mu = sum(data) / len(data)
	data -= mu
	return data

def pca(data, dim = 2):
	'''
	returns the components and reduced data
	'''
	n, p = data.shape
	xs = center(data)
	cov = np.zeros((p,p))
	components = np.zeros((p, dim))

	for i in range(len(xs)):
		cov += np.dot(xs[i].reshape([p,1]),xs[i].reshape([p,1]).T)
	cov /= n

	eigvals, eigvecs = np.linalg.eig(cov)

	for i in range(dim):
		argmax = eigvals.argmax()

		components[:,i] = eigvecs[:,argmax]
		eigvals = np.delete(eigvals, argmax)
		eigvecs = np.delete(eigvecs, argmax, axis = 1)

	return np.dot(data, components)