# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:40:20 2020

@author: samuel stalder
"""
import numpy as np

e = np.exp;

def f(x):
    return np.exp((x**2)) + x**-3 -10

def f_diff(x):
    return 2 * np.exp((x**2)) * x - (3/(x**4))

def d(h):
    return (np.pi/3) * h**2 * (15-h) - 471

def d_diff(h):
    return np.pi*(h-10)*h


def sekanten_verfahren(f, x0, x1, tol):
    while np.abs(x0 - x1) > tol:
        x2 = x1 - ((x1 - x0) / (f(x1) - f(x0))) * f(x1)
        print(x2)
        x0 = x1
        x1 = x2


#Aufgabe 1
sekanten_verfahren(f, -1.0, -1.2, 10**-3)    
    
#Aufgabe 2
# sekanten_verfahren(d, 9, 8, 10**-3)


def newton_verfahren(f, f_diff, x0, tol):
    temp = 0
    while np.abs(x0 - temp) > tol:    
        x1 = x0 - f(x0) / f_diff(x0)
        temp = x0
        x0 = x1
        print(x1)
        
def vereinfachtes_newton_verfahren(f, f_diff, x0, tol):
    temp = 0
    x0_base = x0
    while np.abs(x0 - temp) > tol:    
        x1 = x0 - f(x0) / f_diff(x0_base)
        temp = x0
        x0 = x1
        print(x1)


vereinfachtes_newton_verfahren(f, f_diff, 0.5, 10**-3)

#Beim Newton-Verfahren und dem vereinfachtem Newton-Verfahren müsste man die Funktion noch ableiten
