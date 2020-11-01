import numpy as np

def IT19a_ZH6_S6_Aufg2(A,b):
    rows = A[:,0]
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
            A[j] = A[j] - A[i].dot(A[j][i]/A[i][i])
    
    print(A)
    einsetzen(A)

def einsetzen(A):
    rows = len(A)
    columns = len(A[0])
    print(rows)
    print(columns)

    x = []
    for r in range(rows-1, -1, -1):
        offset = abs(rows -1 - r)
        sum = 0
        for c in range(offset):
            sum += x[c] * A[r][-1 - offset + c]
        x.insert(0, (A[r][-1] - sum)/A[r][-2 - offset])
    print(x)

test = np.array([[20,10,0,150],[50,30,20,470],[200,150,100,2150]])
test = np.array([[20,10,0],[50,30,20],[200,150,100]])
b = np.array([[150], [470], [2150]])
IT19a_ZH6_S6_Aufg2(test, b)