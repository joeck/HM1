import numpy as np

A = np.array([[10**-5, 10**-5], [2.,3]])
bt = np.array([[1e-5, 1]]).T
Ai = np.linalg.inv(A)
print(A)
print(np.linalg.norm(A, 1))
print(np.linalg.norm(Ai, 1))
print(np.linalg.norm(A, 1) * np.linalg.norm(Ai, 1))
print(Ai)

x = np.linalg.solve(A, bt)

print(x)