import numpy as np
def gauss(a):
    n = a.shape
    n = n[0]
    x = np.zeros(n)
    for i in range(n):
        print(i)
        if a[i][i] == 0.0:
            break
        for j in range(i+1, n):
            ratio = a[j][i]/a[i][i]
            for k in range(n+1):
                a[j][k] = a[j][k] - ratio * a[i][k]
    # Back Substitution
    x[n-1] = a[n-1][n]/a[n-1][n-1]
    for i in range(n-2,-1,-1):
        x[i] = a[i][n]
        for j in range(i+1,n):
            x[i] = x[i] - a[i][j]*x[j]
        x[i] = x[i]/a[i][i]
    print("My solution: ",x)
    a = np.delete(a,3,axis=1)
    print("The new matrix: \n",a)
    det = 1
    for x in range(0,n):
        det = det * a[x,x]
    print("Determinante: ",det)
matrix = np.array([[20,10,0,150],[50,30,20,470],[200,150,100,2150]])
gauss(matrix)


#=======================3===========================

A1 = np.array([[4,-1,-5,-5],[-12,4,17,19],[32,-10,-41,-39]])
a1 = np.delete(A1,3,axis = 1)
b1 = A1[:,3]
gauss(A1)
solved = np.linalg.solve(a1,b1)
print("Numpy sais: ",solved)

A2 = np.array([[2,7,3,25],[-4,-10,0,-24],[12,34,9,107]])
a2 = np.delete(A2,3,axis = 1)
b2 = A2[:,3]
gauss(A2)
solved = np.linalg.solve(a2,b2)
print("Numpy sais: ",solved)

A3 = np.array([[-2,5,4,1],[-14,38,22,40],[6,-9,-27,75]])
a3 = np.delete(A3,3,axis = 1)
b3 = A3[:,3]
gauss(A3)
solved = np.linalg.solve(a3,b3)
print("Numpy sais: ",solved)

A4 = np.array([[-1,2,3,2,5,4,3,-1,-11],[3,4,2,1,0,2,3,8,103],[2,7,5,-1,2,1,3,5,53],[3,1,2,6,-3,7,2,-2,-20],[5,2,0,8,7,6,1,3,95],[-1,3,2,3,5,3,1,4,78],[8,7,3,6,4,9,7,9,131],[-3,14,-2,1,0,-2,10,5,-26]])
a4 = np.delete(A4,8,axis = 1)
b4 = A4[:,7]
gauss(A4)
solved = np.linalg.solve(a4,b4)
print("Numpy sais: ",solved)