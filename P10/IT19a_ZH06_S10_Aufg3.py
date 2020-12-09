# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:08:36 2020

@author: samuel
"""
import numpy as np

A = np.array([[8,5,2],[5,9,1],[4,2,7]])
b = np.array([19,5,34])
x0 = np.array([1,-1,3])
tol = 1/10000
opt = False

def F_jacobi(x, L, D, R):
    return -np.linalg.inv(D) @ (L+R) @ x + np.linalg.inv(D) @ b

def F_gauss_seidel(x, L, D, R):
    return -np.linalg.inv(D + L) @ R @ x + np.linalg.inv(D + L) @ b


# opt: 
#   true: jacobi
#   false: gauss seidel
def Stalder_Samuel_6_S10_Aufg3a(A,b,x0,tol,opt):
    
    #diagonal matrix von A
    D = np.diag(np.diag(A)) 
    
    R = np.triu(A) - D
    L = np.tril(A) - D
    # startwert x
    x = x0
    
    a_posteriori = 1
    counter = 0
    
    while(a_posteriori > tol):
        if(opt):
            x_next = F_jacobi(x, L, D, R)
        else:
            x_next = F_gauss_seidel(x, L, D, R)
        # print(x)
        a_posteriori = np.linalg.norm(x - x_next, np.inf)
        # print(a_posteriori)
        x = x_next
        counter += 1
    # print(counter)
    xn = x_next
    n = counter
    n2 = counter
    
    return [xn, n, n2]

print(Stalder_Samuel_6_S10_Aufg3a(A,b,x0,tol,opt))