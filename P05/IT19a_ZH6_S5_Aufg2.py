import numpy as np

maxV = 471
r = 5

def h(x):
    return np.sqrt((maxV * 3)/(np.pi * (3*r - x)))

x = 8.995
before = 0
count = 0
while (abs(x-before) > 10**-3 and count < 50):
    print(x)
    before = x
    x = h(x)
    count += 1
print("Final: " + str(x))