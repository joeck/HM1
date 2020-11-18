import numpy as np

A = np.array([[1.,-2,3], [-5,4,1], [2,-1,3]])
b = np.array([1.,9,5]).reshape(3,1)
u1 = np.array([0.769, -0.594, 0.237]).reshape((3,1))

h1 = np.eye(3, dtype=float) - 2 * np.matmul(u1, np.transpose(u1))
# print(np.matmul(h1,A))

A2 = np.matmul(h1,A)[1:, 1:]
# print(A2)
# print(A2[:,0].reshape(2,1).shape[0])
# print(np.linalg.norm(A2[:,0],2) * np.array([1,0]).reshape((2,1)) + A2[:,0].reshape(2,1))

def QR(AA, b):
    A = np.copy(AA)
    Ao = np.copy(AA)
    b = np.copy(b)
    Qs = []

    cols = A.shape[0]
    for i in range(cols, 1, -1):
        print(i)
        col1 = A[:,0].reshape(i,1)
        v1 = col1 + sign(A[0][0]) * np.linalg.norm(col1,2) * e(i)
        u1 = v1/np.linalg.norm(v1,2)
        H1 = np.eye(i, dtype=float) - (np.multiply(2,np.dot(u1, u1.T)))
        Q = padding(H1, Ao)
        Qs.append(Q)
        print("H1", H1)
        print("Q",Q)
        print("Ao", Ao)
        print("Ai", np.dot(Q, AA))
        A = np.dot(Q,Ao)[1:,1:]
        print("A",A)
    print(Qs)

    #R = 
    R = np.eye(Ao.shape[0], dtype=float)
    Q = np.eye(Ao.shape[0], dtype=float)
    for val in Qs:
        print(val)
        print(val.T)
        Q = np.dot(Q, val.T)
        R = np.dot(R, val)
        print("Rloop", R)

    R = np.dot(R,Ao)
    print("R",R)
    print("Q",Q)
    
    print("A",np.dot(Q,R))


def sign(x):
    if x < 0:
        return -1
    return 1

def e(x):
    e = np.zeros(x, dtype=float).reshape(x, 1)
    e[0][0] = 1
    return e

def padding(a, size):
    if a.shape == size.shape: return a
    result = np.zeros_like(size, dtype=float)
    xo = size.shape[0] - a.shape[0]
    result[xo:a.shape[0]+xo, xo:a.shape[1]+xo] = a
    for i in range(0, xo, 1): result[i][i] = 1
    return result


At = np.array([[1.,2,-1], [4,-2,6],[3,1,0]])
bt = np.array([9,-4,9]).reshape(3,1)

QR(At, bt)
#t = np.array([[-0.91387533,  1.40599493],[ 0.40599493, -0.08612467]])
#print(padding(t,A))