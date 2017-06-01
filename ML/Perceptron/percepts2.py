import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial
import time
np.random.seed(17)

def distance(x1,x2):
	return scipy.spatial.distance.sqeuclidean(x1, x2)

def gauss(x1, x2, sigma = 1):
	return np.exp(-distance(x1, x2) / 2 / sigma ** 2)

def linear(x1, x2, sigma = 1):
	return np.dot(x1, x2)

def learn_linear_perceptron(xs, ys, w = np.array([0])):
	row, col = xs.shape
	yhat = np.zeros(row)
	if len(w) == 1:
		w = np.zeros(col)
	for t in range(row):
		if np.dot(w, xs[t]) >= 0:
			yhat[t] = 1
		else:
			yhat[t] = -1
		if yhat[t] == -1 and ys[t] == 1:
			w += xs[t]
		if yhat[t] == 1 and ys[t] == -1:
			w += -xs[t]
	return w, yhat

def batch_linear_perceptron(xs, ys):
	row, col = xs.shape
	yhat = np.zeros(row)
	w = np.array([0])
	error_rate = 0.01
	max_iter = 500
	t = 0
	preds = 1 + ys
	while t < max_iter and np.sum(ys != preds) / len(ys) > error_rate:
		w, preds = learn_linear_perceptron(xs, ys, w)
		t += 1

	print("Batch completed in {} iterations with an error rate of {}".format(t, np.sum(ys != preds) / len(ys)))
	return w, preds

def learn_kernel_perceptron_online(xs, ys, kernel, sigma = 1, c = np.array([0])):
	'''
	x is a t by 1 array
	'''
	row, col = xs.shape
	yhat = np.zeros(row)
	if len(c) == 1:
		c = np.zeros(row)
	for t in range(0,row):
		if t == 0:
			estimate = 0
		else:
			estimate = np.sum(c[0:t] * np.apply_along_axis(kernel, 1, xs[0:t], xs[t]))

		if estimate >= 0:
			yhat[t] = 1
		else:
			yhat[t] = -1

		if yhat[t] == -1 and ys[t] == 1:
			c[t] = 1
		elif yhat[t] == 1 and ys[t] == -1:
			c[t] = -1
	return yhat, c

def learn_kernel_perceptron_batch(xs, ys, kernel, sigma = 1, c = np.array([0])):
	'''
	x is a t by 1 array
	'''
	row, col = xs.shape
	yhat = np.zeros(row)
	if len(c) == 1:
		c = np.zeros(row)
	for t in range(row):
		estimate = 0
		for i in range(row):
			estimate += c[i] * kernel(xs[i], xs[t], sigma)
		if estimate >= 0:
			yhat[t] = 1
		else:
			yhat[t] = -1

		if yhat[t] == -1 and ys[t] == 1:
			c[t] += 1
		elif yhat[t] == 1 and ys[t] == -1:
			c[t] += -1

	w = np.zeros(col)
	for i in range(row):
		w += c[i] * xs[i]

	return w, yhat, c

def batch_kernel_perceptron(xs, ys, kernel, sigma = 1):
	'''
	c doesn't seem to change
	'''
	row, col = xs.shape
	yhat = np.zeros(row)
	c = np.array([0])
	error_rate = 0.01
	max_iter = 500
	t = 0
	preds = 1 + ys
	while t < max_iter and np.sum(ys != preds) / len(ys) > error_rate:
		print("Batch iteration {}, {}".format(t, np.sum(ys != preds) / len(ys)))
		w, preds, c = learn_kernel_perceptron_batch(xs, ys, kernel, c = c)
		t += 1

	print("Batch completed in {} iterations with an error rate of {}".format(t, np.sum(ys != preds) / len(ys)))
	return preds, c

def test_linear_perceptron(xs, w):
	row, col = xs.shape
	yhat = np.zeros(row)
	for t in range(row):
		if np.dot(w, xs[t]) >= 0:
			yhat[t] = 1
		else:
			yhat[t] = -1

	return yhat

def test_kernel_perceptron(remembered, xs, kernel, c, sigma = 1):
	row, col = xs.shape
	row1, col1 = remembered.shape
	yhat = np.zeros(row)
	for t in range(row):
		estimate = 0
		for i in range(row1):
			estimate += c[i] * kernel(remembered[i], xs[t], sigma)
		if estimate >= 0:
			yhat[t] = 1
		else:
			yhat[t] = -1
	return yhat

def np_data(data):
	data = np.loadtxt(data)
	return data

def test_linear():
	w, preds = learn_linear_perceptron(xs,ys)
	tpreds = test_linear_perceptron(testing, w)
	np.save("{}kernelonline_l".format('linear'), tpreds)

	w2, preds2 = batch_linear_perceptron(xs, ys)
	tpreds2 = test_linear_perceptron(testing, w2)
	np.save("{}kernelbatch_l".format('linear'), tpreds2)

	print(np.sum(ys == preds) / len(ys))
	print(np.sum(ys == preds2) / len(ys))
	return preds, preds2

def test_rbf(k = gauss):
	preds, c = learn_kernel_perceptron_online(xs, ys, k, sigma = 2.5)
	tpreds = test_kernel_perceptron(xs, testing, gauss, c, sigma = 2.5)
	np.save("{}kernelonline_l".format('gauss'), tpreds)

	preds2, c2 = batch_kernel_perceptron(xs, ys, k)
	tpreds2 = test_kernel_perceptron(xs, testing, k, c2, sigma = 2.5)
	np.save("{}kernelbatch_l".format('gauss'), tpreds2)

	print(np.sum(ys == preds) / len(ys))
	print(np.sum(ys == preds2) / len(ys))
	return preds, preds2

def cross(xs, ys, s_list, k = 3):
	row, col = xs.shape

	one = False
	max_iter = 5
	errors = []
	arr = np.arange(row) % k
	np.random.shuffle(arr)

	for s in s_list:
		print("Testing {}".format(s))
		holdout = []

		for i in range(k):
			preds, c = learn_kernel_perceptron_online(xs[arr == i], ys[arr == i], gauss, sigma = s)
			preds = test_kernel_perceptron(xs[arr == i], testing, gauss, c, sigma = s)
			holdout.append(np.sum(test_labs == preds) / len(test_labs))
		errors.append(sum(holdout) / len(holdout))

	return s_list[errors.index(max(errors))], errors, s_list

def grapher(x, y, main = '', xlab = '', ylab = ''):
	fig, ax = plt.subplots()
	ax.plot(x, y, marker = 'o', markersize = 3)
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.title(main)
	plt.show()
	plt.clf()

#Writeup a
def wa():
	linears = []
	kernels = []
	values = np.arange(50, 2001, 50)
	for i in values:
		print(i)
		w, preds = learn_linear_perceptron(xs[:i],ys[:i])
		linears.append(np.sum(ys[:i] == preds) / len(ys[:i]))

		preds, c = learn_kernel_perceptron_online(xs[:i], ys[:i], gauss, sigma = 2.7)
		kernels.append(np.sum(ys[:i] == preds) / len(ys[:i]))
	np.save('linears_multi', linears)
	np.save('kernels_multi', kernels)

#Writeup b
def wb():
	'''
	Assumes the cross has already been run
	'''
	y = np.load('errors.npy')
	x = np.load('s_list.npy')
	grapher(x, y, "Cross-Validation Accuracy vs Sigma", "Sigma", "Accuracy")

xs = np_data('C:/Users/Phil/Downloads/train2k.databw.35')[0:2000,:]
ys = np_data('C:/Users/Phil/Downloads/train2k.label.35')[0:2000]
testing = np_data('C:/Users/Phil/Downloads/test200.databw.35')
test_labs = np_data('C:/Users/Phil/Downloads/test200.label.35')

# s_list = np.append(np.arange(.1, 1.01, .1), np.array([1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]))
# s_list = np.arange(2, 3, .05)

# sigma, errors, s_list = cross(xs, ys, s_list)
# np.save('errors23', errors)
# np.save('s_list23', s_list)

# wa()
# wb()

# x = np.arange(50, 2001, 50)
# y1 = np.load('linears_multi.npy')
# y2 = np.load('kernels_multi.npy')

# grapher(x, y1, "Mistakes vs # of Examples Seen, Linear Online", "Examples Seen", "Accuracy")
# grapher(x, y2, "Mistakes vs # of Examples Seen, RBF Online", "Examples Seen", "Accuracy")

test_linear()
test_rbf()

# np.savetxt('linearonline.label.35', np.load('linearkernelonline_l.npy'), fmt='%i')
# np.savetxt('linearbatch.label.35', np.load('linearkernelbatch_l.npy'), fmt='%i')
# np.savetxt('gaussonline.label.35', np.load('gausskernelonline_l.npy'), fmt='%i')
# np.savetxt('gaussbatch.label.35', np.load('gausskernelbatch_l.npy'), fmt='%i')
# import os
# print(os.path.dirname(os.path.dirname(__file__)))   np.sum(ys[:i] == preds) / len(ys[:i]))

# print(ys.shape)
# print(np.load('linearkernelonline_l.npy').shape)
# print("linear kernel online", np.sum(ys == np.load('linearkernelonline_l.npy') / len(ys)))
# print("linear kernel batch", np.sum(ys == np.load('linearkernelbatch_l.npy') / len(ys)))
# print("kernel kernel online", np.sum(ys == np.load('gausskernelonline_l.npy') / len(ys)))
# print("kernel kernel batch", np.sum(ys == np.load('gausskernelbatch_l.npy') / len(ys)))
