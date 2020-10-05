import numpy as np

A = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
B = np.array([[5, 4, 3, 2], [4, 3, 2, 5], [3, 2, 5, 4], [2, 5, 4, 3]])
b = np.array([1, 2, 3, 4])

print(A.dot(b))
print(B.dot(b))
print(A.T)
print(B.T)
print(np.dot(A.T, A))
print(np.dot(B.T, B))

print(A[4:4, :])