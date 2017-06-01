import numpy as np
import pandas as pd

pij=pd.DataFrame(np.random.randn(500,2))
# print(pij.head())
pij = pij.divide(pij.sum(1), axis=0) #
print(pij.head())
# print(pij.apply(, axis=0, reduce = False, args = (pij.sum(1))))

def eric_divide(row):
	return row / row.sum(1)


# print(np.divide(pij,4))
def onee(x,y):
	# print(x,y)
	return np.divide(x,y)

print(pij.apply(np.divide, axis = 0, args = (pij.sum(1),)))