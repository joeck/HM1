import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.arange(6, 1000, 6)
def ss(n):
    if (n <= 6):
        return 1
    return np.sqrt(2 - (2 * np.sqrt(1 - (pow(s(n/2),2)/4))))

#y = [i*s(i) for i in x]
y = []
n = 6
s = 1
for i in range(30):
    s = np.sqrt(2 - 2 * np.sqrt(1 - (s ** 2 / 4)))
    n = n * 2
    y.append(s / 2 * n)

print(y)
ax.plot(y, label="S(n)")

