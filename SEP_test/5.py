import numpy as np

A = np.array([[240, 120, 80], [60, 180, 170], [60, 90, 500]])
b = np.array([[3080, 4070, 5030]]).T

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
        #print(A)
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
    return x

#determinante
def determinante(A):
    det = 1
    for i in range(len(A[0])):
        det *= A[i][i]
    return det

def GaussVerfahren(A,b):
    oDM = obereDreiecksMatrix(A, b)
    x = einsetzen(oDM)
    aODM = oDM[:,:-1]

    print("Dreiecksmatrix: ")
    print(aODM)
    print("Solution: ")
    print(x)

print(GaussVerfahren(A, b))

#c)
# A = np.array([[240, 120, 80], [60, 180, 170], [60, 90, 500]])
# b = np.array([[3080, 4070, 5030]]).T
bd = 0.95*b
deltaB = np.subtract(b, bd)
err = np.linalg.norm(np.linalg.inv(A), np.inf) * np.linalg.norm(deltaB, np.inf)

print(np.linalg.norm(np.linalg.inv(A), np.inf))
print(np.linalg.norm(deltaB, np.inf))
print(np.linalg.norm(b, np.inf))
print(err)

#d)
print("Kondition")
condA = np.linalg.norm(A, np.inf) * np.linalg.norm(np.linalg.inv(A), np.inf)
print(condA)