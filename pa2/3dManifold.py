'''
this assumes that the last column is a color coding
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import PCA
import Isomap
import LLE
import LaplacianEigenmap
from mpl_toolkits.mplot3d import Axes3D

#Data parsing
data = pd.read_csv("3Ddata.txt", sep=r"\s+", header = None)
n, p = data.shape
npdata = np.array([data.ix[:,i] for i in range(p-1)]).transpose()
color_code = np.array(data.ix[:, p-1]).reshape([n, 1])

#make general
color_converter = {1: 'green', 2: 'red', 3: 'black', 4: 'blue'}

def color_convert(num):
	return color_converter[num]

def graph3d(data, display = True, file_name = None, verbose = True):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')

	for code in np.unique(data[:,3]):
		x, y, z = zip(*data[data[:,3] == code][:,0:3])
		ax.scatter(x, y, z, c = color_convert(code), marker = 'o')

	if file_name != None:
		fig.savefig(file_name)
	if display == True:
		plt.show()
	plt.clf()

def graph2d(data, display = True, file_name = None, verbose = True):
	fig, ax = plt.subplots()

	for code in np.unique(data[:,2]):
		x, y = zip(*data[data[:,2] == code][:,0:2])
		ax.scatter(x, y, c = color_convert(code), marker = 'o')

	if file_name != None:
		fig.savefig(file_name)
	if display == True:
		plt.show()
	plt.clf()

'''
Examples, in order, 3d plot of data, PCA, Isomap, LLE, LapEig
'''
# graph3d(np.column_stack((npdata, color_code)))
graph2d(np.column_stack((PCA.pca(npdata, dim = 2), color_code)), False, 'PCA')
graph2d(np.column_stack((Isomap.isomap(npdata, load = 'C.npy'), color_code)), False, 'Isomap')
graph2d(np.column_stack((LLE.lle(npdata), color_code)), False, 'LLE')
graph2d(np.column_stack((LaplacianEigenmap.le(npdata), color_code)), False, 'LaplacianEigenmap')

#Just a sanity check
# from sklearn import manifold
# x = manifold.SpectralEmbedding().fit_transform(X= npdata)
# graph2d(np.column_stack((x, color_code)))