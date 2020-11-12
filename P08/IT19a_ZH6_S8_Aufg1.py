import numpy as np

A = np.array([[1,-2,3], [-5,4,1], [2,-1,3]])
b = np.array([1,9,5]).reshape(3,1)
u1 = np.array([0.769, -0.594, 0.237]).reshape((3,1))

h1 = np.eye(3, dtype=float) - 2 * np.matmul(u1, np.transpose(u1))
# print(np.matmul(h1,A))

A2 = np.matmul(h1,A)[1:, 1:]
# print(A2)
# print(A2[:,0].reshape(2,1).shape[0])
# print(np.linalg.norm(A2[:,0],2) * np.array([1,0]).reshape((2,1)) + A2[:,0].reshape(2,1))

QR(A,b)

def QR(A, b):
    A = np.copy(A)
    b = np.copy(b)

    cols = A.shape[0]
    col1 = A[:,0].reshape(cols,1)
    v1 = col1 + sign(A[0][0]) * np.linalg.norm(col1) * e(cols)
    print(v1)


def sign(x):
    if x < 0:
        return -1
    return 1

def e(x):
    e = np.zeros(x).reshape(x, 1)
    e[0][0] = 1
    return e
