import numpy as np

def IT19a_ZH6_S6_Aufg2(A,b):
    oDM = obereDreiecksMatrix(A, b)
    x = einsetzen(oDM)
    aODM = oDM[:,:-1]
    det = determinante(aODM)

    print("\n-------------")
    print("Dreiecksmatrix: \n" + str(aODM))
    print("Determinante: " + str(det))
    print("Solution: " + str(x))

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
    return A

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

def determinante(A):
    det = 1
    for i in range(len(A[0])):
        det *= A[i][i]
    return det


# test = np.array([[20,10,0],[50,30,20],[200,150,100]])
# b = np.array([[150], [470], [2150]])
# IT19a_ZH6_S6_Aufg2(test, b)

# A1 = np.array([[4,-1,-5,-5],[-12,4,17,19],[32,-10,-41,-39]])
# a1 = np.delete(A1,3,axis = 1)
# b1 = A1[:,3]
# IT19a_ZH6_S6_Aufg2(a1, np.vstack(b1))
# solved = np.linalg.solve(a1,b1)
# print("Numpy sais: ",solved)