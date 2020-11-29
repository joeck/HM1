# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:30:11 2020

@author: samuel
"""


from matplotlib.pyplot import semilogy
from IT19a_ZH6_S9_Aufg2 import Gruppe_6_S9_Aufg2 as aufgabe2
import numpy as np
import matplotlib.pyplot as plt

y_max = []
y_obs = []
y_rel = []

for i in range(0, 1000):
    
    A = np.random.rand(100, 100)
    b = np.random.rand(100)
    A_g = A + np.random.rand(100, 100) / 10**5
    b_g = b + np.random.rand(100, 1) / 10**5

    [_, _, dx_max, dx_obs] = aufgabe2(A, A_g, b, b_g)
    relation = dx_max / dx_obs
    
    y_max.append([dx_max])
    y_obs.append([dx_obs])
    y_rel.append([relation])
    

x = np.arange(1, 1001)

plt.plot(x, y_max, label="max")
plt.plot(x, y_obs, label="obs")
plt.plot(x, y_rel, label="rel")



plt.grid()
plt.legend()
plt.semilogy()
plt.show()

"""
dx_max ist keine realistische obere Schranke
"""