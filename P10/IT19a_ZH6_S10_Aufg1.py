import numpy as np
from numpy.linalg import inv

A = np.array([[8,5,2],[5,9,1],[4, 2, 7]])
L = np.array([[0,0,0],[5,0,0],[4, 2, 0]])
D = np.array([[8,0,0],[0,9,0],[0, 0, 7]])
R = np.array([[0,5,2],[0,0,1],[0, 0, 0]])

invD = inv(D.copy())

b = np.array([19,5,34])

xZero = np.array([1,-1,3])

def jacobi(x):
    return -invD.dot(L+R).dot(x) + invD.dot(b)

xOne = jacobi(xZero)
xTwo = jacobi(xOne)
xThree = jacobi(xTwo)

print("x1: ", xOne)
print("x2: ",xTwo)
print("x3: ",xThree)
