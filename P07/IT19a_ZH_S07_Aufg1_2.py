import numpy as np
from scipy.linalg import lu

def gauss(a):
    n = a.shape
    n = n[0]
    x = np.zeros(n)
    for i in range(n):
        if a[i][i] == 0.0:
            break
        for j in range(i + 1, n):
            ratio = a[j][i] / a[i][i]
            for k in range(n + 1):
                a[j][k] = a[j][k] - ratio * a[i][k]
    # Back Substitution
    x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = a[i][n]
        for j in range(i + 1, n):
            x[i] = x[i] - a[i][j] * x[j]
        x[i] = x[i] / a[i][i]
    print("My solution: ", x)
    a = np.delete(a, 3, axis=1)
    print("The new matrix: \n", a)
    det = 1
    for x in range(0, n):
        det = det * a[x, x]
    print("Determinante: ", det)


matrix = np.array([[20,30,10,520], [10,17,6,300], [2,3,2,76]])
gauss(matrix)


matrix2 = np.array([[0.8,2.2,3.6],[2.0,3.0,4.0],[1.2,2.0,5.8]])
print("LU:")
print(lu(matrix2))