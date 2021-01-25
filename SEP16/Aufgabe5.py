import numpy as np

def F_jacobi(x, b, L, D, R):
    return -np.linalg.inv(D) @ (L+R) @ x + np.linalg.inv(D) @ b

a = 40
L = np.array([
    [0,0,0],
    [10,0,0],
    [5,20,0]
], dtype=np.float64)

D = np.array([
    [30,0,0],
    [0,a,0],
    [0,0,50]
], dtype=np.float64)

R = np.array([
    [0,10,5],
    [0,0,20],
    [0,0,0]
], dtype=np.float64)

b = np.array([[5*a, a, 5*a]]).T
x0 = np.array([[a, 0, a]]).T

x1 = F_jacobi(x0, b, L, D, R)
print(x1)