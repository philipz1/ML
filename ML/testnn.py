import numpy as np

def nonlin(x, deriv = False):
	if deriv == True :
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

def neur(Xs, Ys, layers = 3):
	layers = layers - 1
	x0, x1 = Xs.shape
	weights = [0] * layers
	weights[0] = 2 * np.random.random((x1, x0)) - 1
	for layer in range(1,layers - 1):
		weights[layer] = 2 * np.random.random((x0, x0)) - 1
	weights[-1] = 2 * np.random.random((x0, 1)) - 1

	layer_values = [0] * (layers + 1)
	errors = [0] * layers
	deltas = [0] * layers
	layer_values[0] = Xs
	print(weights)
	for j in range(10000):

		# Feed forward through layers
		for layer in range(1, layers + 1):
			layer_values[layer] = nonlin(np.dot(layer_values[layer - 1], weights[layer - 1].T))


		for i in range(1, layers + 1):
			if i == 1:
				errors[-i] = Ys - layer_values[-i]
			else:
				errors[-i] = deltas[-i + 1].T.dot(weights[-i+1])
			deltas[-i] = errors[-i] * nonlin(layer_values[-i], deriv = True)

		if (j % 1000) == 0:
			print ("Error:" + str(np.mean(np.abs(errors[-1]))))

		for i in range(1, layers + 1):
			weights[-i] += layer_values[-i - 1].T.dot(deltas[-i])

	# for i in range(len(layer_values)):
	# 	print(layer_values[i])
	return layer_values

X = np.array([np.random.randint(2, size = 1) for i in range(4)])
print(X.shape)
Y = np.array(np.random.randint(2, size = 4)).reshape((4,1))
print(Y)

print(neur(X,Y,2))