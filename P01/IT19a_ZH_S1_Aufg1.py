import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 11, 0.001)
fx = pow(x, 5) - 5 * pow(x, 4) - 30 * pow(x, 3) + 110 * pow(x, 2) + 29 * x - 105
f_x = 5 * x**4 - 20 * x ** 3 - 90 * x ** 2 + 220 * x + 29
Fx = x**6 / 6 - x**5 - 15 * x**4 / 2 + 110 * x**3 / 3 + 29 * x**2/ 2 - 105 * x
plt.plot(x, fx, label="Polynom")
plt.plot(x, f_x, label="Ableitung")
plt.plot(x, Fx, label="Stammfunktion")
plt.xlim(-10 , 10)
plt.ylim(-2000, 2000)
plt.grid()
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Aufgabe 1")
plt.legend()
plt.show()

def horner(p, x):
    y = p[0]*np.ones(x.shape, dtype=np.float64)
    for k in range(1, p.size):
        y = y*x + p[k]*np.ones(x.shape, dtype=np.float64)
    return y