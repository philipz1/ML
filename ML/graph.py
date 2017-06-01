'''
For plotting 2d, 3d clustering
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import kMeans
import kMixture
# import pa2.PCA
from mpl_toolkits.mplot3d import Axes3D

# matplotlib.style.use('ggplot')
# pd.set_option('display.width', 175)
toydata = pd.read_csv("Data/toydata.txt", sep=r"\s+", header = None)
toydata = np.array([toydata.ix[:,0],toydata.ix[:,1]]).transpose()

def get_Bounds(data):
	'''
	gets the xlim and ylim for the scaling parameters
	'''
	p = len(data[list(data.keys())[0]][0])
	bounds = np.zeros([p, p]) #columns are dimensions/axes

	for key in data.keys():
		for i in range(p):
			if min([j[i] for j in data[key]]) < bounds[i, 0]:
				bounds[i, 0] = min([j[i] for j in data[key]])
			if max([j[i] for j in data[key]]) > bounds[i, 1]:
				bounds[i, 1] = max([j[i] for j in data[key]])

	for i in range(len(bounds)):
		for a in range(p):
			bounds[i][a] = bounds[i][a] * 1.1

	return bounds

def graph2d(data, display = True, file_name = None, verbose = True):
	print(data)
	fig, ax = plt.subplots()

	bounds = get_Bounds(data)

	for key in data.keys():
		temp = pd.DataFrame(data[key])
		temp.columns = ['x','y']
		temp.plot(x = 'x', y = 'y', style = 'p', legend = False, xlim = bounds[0,:], ylim = bounds[1,:], ax = ax)

	data_x = []
	data_y = []
	for key in data.keys():
		data_x.append(key[0])
		data_y.append(key[1])
	ax.plot(data_x, data_y, marker = 'o', color = 'white', linestyle = 'None', markersize = 10)
	
	if file_name != None:
		fig.savefig(file_name)
	if display == True:
		plt.show()
	plt.clf()

def graph3d(data, display = True, file_name = None, verbose = True):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# bounds = get_Bounds(data)

	for key in data.keys():
		x, y, z = zip(*data[key])
		ax.scatter(x, y, z, c = np.random.rand(4,), marker = 'o')

	# plt.show()
	centers = [0] * len(key)
	for key in data.keys():
		for i in range(len(centers)):
			centers[i] = key[i]
		ax.scatter(centers[0], centers[1], centers[2], marker = 'o', c = 'black')
	
	if file_name != None:
		fig.savefig(file_name)#, bbox_inches='tight
	if display == True:
		plt.show()
	plt.clf()

def graph_it(data, display = True, file_name = None, verbose = True):
	'''
	The data should be in the form of a dictionary with centers as keys and values arrays
	file_name, if passed, will write the graph to the output file
	display, if True, will display the graph upon completion
	'''
	p = len(data[list(data.keys())[0]][0])
	if p == 2:
		return graph2d(data, display, file_name, verbose)
	elif p == 3:
		graph3d(data, display, file_name, verbose)

#Examples
graph_it(kMeans.kmeans(toydata, 3, plus = False))
# graph_it(kMixture.kmix(toydata, 3, init = 'kmeans'), file_name = 'lol.png')
# graph_it(kMeans.kmeans(np.random.rand(1000,2), 8, plus = True))


# toydata = pd.read_csv("Data/3Ddata.txt", sep=r"\s+", header = None)
# print(toydata)
# toydata = np.array([toydata.ix[:,0],toydata.ix[:,1], toydata.ix[:,2]]).transpose()
# graph3d(kMeans.kmeans(toydata, 3, plus = True))

def clusterer(n, clusters):
	per = n // clusters
	final = np.array([0,0])
	for i in range(clusters):
		center = np.random.rand(1,2)
		dists = np.random.rand(per, 1)
		offsets = np.random.randint(0, 359) / 2 / np.pi
		pointsx = center[:,0] + dists[:,0] * np.cos(offsets)
		pointsy = center[:,1] + dists[:,1] * np.sin(offsets)
		np.concatenate(pointsx, pointsy, axis = 1)
		np.concatenate(final, center)
		np.concatenate(final, pointsx)

	return final

# clusterer(10, 2)