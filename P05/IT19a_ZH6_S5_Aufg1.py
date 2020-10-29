import numpy as np

def f(x):
    return np.exp(x**2) + x**-3 - 10

def f1(x):
    return 2 * np.exp(x**2) * x - 3/(x**4)

def newton(f, f1, x):
    return x - f(x)/f1(x)

def newton_easy(f, f1, x, x0):
    return x - f(x)/f1(x0)

def sekant(f, x0, x1):
    return x1 - ((x1 - x0)/(f(x1)-f(x0)) * f(x1))

count = 0
x0 = -1.0
x1 = -1.2
while (count < 10 ):
    print("x" + str(count) + ": " + str(x1))
    x = sekant(f, x0, x1)
    x0 = x1
    x1 = x
    count += 1
print(x1)