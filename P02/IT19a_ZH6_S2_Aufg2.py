import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3)
fig.suptitle('Aufgabe 2')

# Aufgabe a
x = np.linspace(1.99, 2.01, 501)
f1 = x**7 - 14 * x**6 + 84 * x**5 - 280 * x**4 + 560 * x**3 - 672 * x**2 + 448 * x - 128
f2 = (x - 2)**7

axs[0].plot(x, f1, label="f1")
axs[0].plot(x, f2, label="f2")
axs[0].legend()
# Das ausgeschriebene Polynom ist ungenau, da die einzelnen Punkten mit gerundeten Werten berechntet werden.

# Aufgabe b
# Nein, nicht stabil
# (10**-14 - -10**-14)/10**-17 =~ 2000
y = np.linspace(-10**-14, 10**-14, 2000)
g = y / (np.sin(1 + y) - np.sin(1))

# Aufgabe C
g1 = y / (2 * np.cos(((1 + y + 1) / 2)) * np.sin((1 + y - 1)/2))

# Der Graph ist 'schöner' und sieht regelmässiger aus.
# instabil. g1(x) mit x -> 0 ist immernoch undefiniert da wir durch 0 teilen. sin((1 + 0 -1) / 2) = sin(0) = 0

axs[1].plot(y, g, label="g(x)")
axs[2].plot(y, g1, label="g1(x)")
axs[1].legend()
axs[2].legend()

