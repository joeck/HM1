import numpy as np

def Plambeck_Joel_6_S5_Aufg3(f, x0, x1, tol):
    count = 0
    while (abs(x0-x1) > tol and count < 10000):
        temp = x1
        x1 = x1 - ((x1 - x0)/(f(x1)-f(x0)) * f(x1))
        x0 = temp
        count += 1
    print(x1)
    return x1

def f(x):
    return np.exp(x**2) + x**-3 - 10

Plambeck_Joel_6_S5_Aufg3(f, -1.0, -1.2, 10**-5)

# Beim newton verfahren und dem vereinfachten newton verfahren wird die Ableitung der Funktion benötigt.