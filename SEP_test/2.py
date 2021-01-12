import numpy as np
import matplotlib.pyplot as plt

def f(x) : return x * np.exp(x)
def df(x): return np.exp(x)*(x+1)  
def K(x) : return np.abs(x)*np.abs(df(x))/np.abs(f(x))  
# K = abs(x + 1)  
x = np.arange(-4,2.05,0.05)
plt.plot(x,K(x),'-'), plt.xlabel('x'), plt.ylabel('K(x)')