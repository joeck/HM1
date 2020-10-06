import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
n = 6
s = 1
for i in range(20):
    s = np.sqrt(2 - 2 * np.sqrt(1 - (s ** 2 / 4)))
    n = n * 2
    y.append(s / 2 * n)
    x.append(n)

plt.plot(x, y)
plt.xlim(1,1000)
plt.ylim(3.1,3.15)
plt.xlabel("Anzahl Ecken des Vielecks")
plt.ylabel("Annäherung an Pi")
plt.grid()

# Grosse n werden wegen falsch gerundet (float)