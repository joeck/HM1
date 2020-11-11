import numpy as np
import matplotlib.pyplot as plt
import IT19a_ZH6_S6_Aufg2 as aufg2

# Use function from last excercise
s7 = np.array([[2,4,8],[9,81,729], [13,169, 2197]])
years = np.array([[-46], [22], [2]])
aufg2.IT19a_ZH6_S6_Aufg2(s7, years)
print(np.linalg.solve(s7, years))

# A
xp = np.array([-0.38250638,7.84249084,-37.15495615, 150])

def poly(i):
    return np.polyval(xp, i)


x = []
y = []

for i in np.arange(0,13, 0.1):
    x.append(i + 1997) #B + 1997
    y.append(poly(i))

plt.plot(x, y, label="Polynom")
plt.xlabel("Jahr")
plt.ylabel("UV indikator")

# C
print("Polynom:")
print("2003 =~ " + str(poly(6)))
print("2004 =~ " + str(poly(7)))

#D
x = np.array([1997,1999,2006,2010])
y = np.array([150,104,172,152])

z = np.polyfit(x, y, 3)
p = np.poly1d(z)

plt.plot(x,y, label="Werte")

x = []
y = []

for i in np.arange(1997,2010, 0.25):
    x.append(i)
    y.append(p(i))

print("Polyfit:")
print("2003 =~ " + str(p(2003)))
print("2004 =~ " + str(p(2004)))

plt.plot(x,y, ".", label="Polyfit")
plt.legend()
