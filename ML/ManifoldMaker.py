import numpy as np

def spiral():
	start = np.array([0,0,0])
	width = 5
	gap = 1

def one_spiral(a, b, theta, points):
	'''
	points is a len n list/array of where we want the values
	'''
	r = np.zeros(len(points))
	for i range(len(points)):
		r[i] = a + b*theta