import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-4, 4, 0.01)
def f(x):
    return x * np.exp(x)

def df(x):
    return (x + 1) * np.exp(x)

def cond(x, f, df):
    return np.abs(df(x)) * np.abs(x) / np.abs(f(x))

plt.plot(x, cond(x, f, df))
