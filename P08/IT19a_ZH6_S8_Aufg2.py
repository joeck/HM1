# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:26:09 2020

Höhere Mathematik 1, Serie 8, Gerüst für Aufgabe 2

Description: calculates the QR factorization of A so that A = QR
Input Parameters: A: array, n*n matrix
Output Parameters: Q : n*n orthogonal matrix
                   R : n*n upper right triangular matrix            
Remarks: none
Example: A = np.array([[1,2,-1],[4,-2,6],[3,1,0]]) 
        [Q,R]=Serie8_Aufg2(A)

@author: knaa
"""
import timeit
import numpy as np
def IT19a_ZH6_S8_Aufg2(A):
    
    
    A = np.copy(A)                       #necessary to prevent changes in the original matrix A_in
    A = A.astype('float64')              #change to float
    
    n = np.shape(A)[0]
    
    if n != np.shape(A)[1]:
        raise Exception('Matrix is not square') 
    
    Q = np.eye(n)
    R = A
    
    for j in np.arange(0,n-1):
        a = np.copy(R[j:,j:(j+1)]).reshape(n-j,1)     
        e = np.eye(n-j, dtype=float)[:,0].reshape(n-j,1)
        length_a = np.linalg.norm(a)
        if a[0] >= 0: sig = 1
        else: sig = -1
        #print("a", a)
        v = a + sig * length_a * e
        #print("v", v)
        u = v / np.linalg.norm(v)
        #print("u", u)
        H = np.eye(n-j, dtype=float) - 2 * u @ np.transpose(u)
        #print("H", H)
        Qi = np.eye(n)
        Qi[j:,j:] = H
        # print("Qi", Qi)
        R = Qi @ R
        Q = Q @ np.transpose(Qi)
        # print("R", R)
        
    return(Q,R)


At = np.array([[1.,2,-1], [4,-2,6],[3,1,0]])
bt = np.array([9,-4,9]).reshape(3,1)
print(IT19a_ZH6_S8_Aufg2(At))
print(np.linalg.qr(At))


A = np.array([
        [1.,-2., 3.],
        [-5., 4., 1.],
        [2., -1., 3.]
        ])

b = np.array([
        [1.],
        [9.],
        [5.]
        ])

t1 = timeit.repeat("IT19a_ZH6_S8_Aufg2(A)", "from __main__ import IT19a_ZH6_S8_Aufg2, A", number=100)
t2 = timeit.repeat("np.linalg.qr(A)", "from __main__ import np, A", number=100)

avg_t1 = np.average(t1)/100
avg_t2 = np.average(t2)/100

print("avg_t1 [A] = ", avg_t1, "\n")
print("avg_t2 [A] = ", avg_t2, "\n")

Test = np.random.rand(100, 100)

t3 = timeit.repeat("IT19a_ZH6_S8_Aufg2(Test)", "from __main__ import IT19a_ZH6_S8_Aufg2, Test", number=100)
t4 = timeit.repeat("np.linalg.qr(Test)", "from __main__ import np, Test", number=100)

avg_t3 = np.average(t3)/100
avg_t4 = np.average(t4)/100

print("avg_t3 [Test] = ", avg_t3, "\n")
print("avg_t4 [Test] = ", avg_t4, "\n")


#
# Feststellung:
# Die linalg.qr ist performanter, besonders bei grossen Matrizen.