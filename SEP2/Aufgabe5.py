import numpy as np
import matplotlib.pyplot as plt

#b
b = np.array([[3080, 4070, 5030]], dtype=np.float64).T

A = np.array([
    [240, 120, 80],
    [60, 180, 170],
    [60, 90, 500]
], dtype=np.float64)

def GaussVerfahren(A,b):
    oDM = obereDreiecksMatrix(A, b)
    x = einsetzen(oDM)
    aODM = oDM[:,:-1]

    print("Dreiecksmatrix: ")
    print(aODM)
    print("Solution: ")
    print(x)
    print("Linalg.solve:")
    print(np.linalg.solve(A, b))

def obereDreiecksMatrix(A, b):
    rows = A[0]
    columns = A[:1,][0]
    A = np.concatenate((A,b), axis=1)
    for i in range(len(columns) - 1):
        if A[i][i] == 0:
            # zeilenvertauschen
            allZeros = True
            for j in range(j + 1, len(rows) + 1):
                if A[j][i] != 0:
                    allZeros = False
                    if j >= i + 1:
                        temp = np.copy(A[i])
                        A[i] = np.copy(A[j])
                        A[j] = np.copy(temp)
            if allZeros:
                raise SystemExit
        for j in range(i+1, len(rows)):
            # eliminationsschritt
            A[j] = A[j] - A[i].dot(A[j][i]/A[i][i])
            print(A)
    return A

#Matrix rückwärts einsetzen
def einsetzen(A):
    rowlength = len(A)
    x = []
    for r in range(rowlength-1, -1, -1): # für jede Zeile beginnend mit letzter
        offset = abs(rowlength -1 - r)
        sum = 0
        for c in range(offset):
            sum += x[c] * A[r][-1 - offset + c] # addieren/einsetzen der schon berechneten unbekannten
        x.insert(0, (A[r][-1] - sum)/A[r][-2 - offset]) # unbekannte der Zeile berechnen
    return np.array([x]).T

GaussVerfahren(A,b)

# [[ 240.  120.   80. 3080.]
#  [   0.  150.  150. 3300.]
#  [  60.   90.  500. 5030.]]

# [[ 240.  120.   80. 3080.]
#  [   0.  150.  150. 3300.]
#  [   0.   60.  480. 4260.]]

# [[ 240.  120.   80. 3080.]
#  [   0.  150.  150. 3300.]
#  [   0.    0.  420. 2940.]]

# Solution:
# [[ 3.]
#  [15.]
#  [ 7.]]

#c

bt = 1.05*b

err_abs = np.linalg.norm(np.linalg.inv(A), np.inf) * np.linalg.norm(b-bt, np.inf)
err_rel = np.linalg.norm(A) * np.linalg.norm(np.linalg.inv(A), np.inf) * np.linalg.norm(b-bt, np.inf) / np.linalg.norm(b, np.inf)

print(err_abs) #2.844345238095238
print(err_rel) #0.35986684399672514

#d
cond = np.linalg.norm(A) * np.linalg.norm(np.linalg.inv(A), np.inf)
cond = np.linalg.cond(A, np.inf)
print(cond) #7.197336879934502