import numpy as np
from numpy import linalg
from numpy.core.numeric import NaN

print("a)\n")

A = np.array([[20_000, 30_000, 10_000],
            [10_000, 17_000, 6_000],
            [2_000, 3_000, 2_000]], dtype=np.float)

b = np.array([[5_720_000], [3_300_000], [836_000]], dtype=np.float)
bg = b + 100_000

[x, xg, dx_max, dx_obs] = Gruppe_6_S9_Aufg2(A, A, b, bg)

print(f"dx max: {dx_max}\ndx obs: {dx_obs}\nx:\n{x}\nx gestört:\n{xg}\n")

#3b
print("="*20)
print("b)\n")

Ag = A + 100

[x, xg, dx_max, dx_obs] = Gruppe_6_S9_Aufg2(A, Ag, b, bg)

print(f"dx max: {dx_max}\ndx obs: {dx_obs}\nx:\n{x}\nx gestört:\n{xg}\n")

#3c
print("="*20)
print("c)\n")

Ag = A - 100

[x, xg, dx_max, dx_obs] = Gruppe_6_S9_Aufg2(A, Ag, b, bg)

print(f"dx max: {dx_max}\ndx obs: {dx_obs}\nx:\n{x}\nx gestört:\n{xg}\n")

def Gruppe_6_S9_Aufg2(A, Ag, b, bg):
    norm = np.linalg.norm

    x = np.linalg.solve(A, b)
    xg = np.linalg.solve(Ag, bg)
    cond_A = np.linalg.cond(A, np.inf)
    
    rel_A = norm(A - Ag, np.inf) / norm(A, np.inf)
    rel_b = norm(b - bg, np.inf) / norm(b, np.inf)

    if cond_A * rel_A < 1:
        dx_max = (cond_A / (1 - (cond_A * rel_A))) * (rel_A + rel_b)
    else:
        dx_max = NaN
    
    dx_obs = norm(x - xg, np.inf) / norm(x, np.inf)
    
    return [x, xg, dx_max, dx_obs]
