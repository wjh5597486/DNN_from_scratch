import numpy as np


x = np.array([
    [0,0],
    [0,1],
    [1,2],
    [3,0],
    [4,1]])
weight = np.array([[1,1,1,1],[1,1,1,1]])
bias = np.ones((1, 4))
print(bias)
xweight = np.dot(x,weight) + bias
print(xweight)