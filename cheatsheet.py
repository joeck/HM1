import numpy as np
from scipy.linalg import lu

#Horner schema
def horner(p, x):
    y = p[0]*np.ones(x.shape, dtype=np.float64)
    for k in range(1, p.size):
        y = y*x + p[k]*np.ones(x.shape, dtype=np.float64)
    return y

#Iterationsverfahren
def f(x):
    return np.exp(x**2) + x**-3 - 10
def f1(x):
    return 2 * np.exp(x**2) * x - 3/(x**4)
count = 0
x0 = -1.0
x1 = -1.2
while (count < 10 ):
    print("x" + str(count) + ": " + str(x1))
    x = sekant(f, x0, x1)
    x0 = x1
    x1 = x
    count += 1
print(x1)

def fixpunktIteraion(f,x0,epsIncr,alpha): #F(x) fixIt gleichung, startpunkt, genauigkeit, lipschitzkonstante 
    import numpy as np
    k=0
    notConverged=True
    N=1000 #max iterationen
    
    while (notConverged and k<N):
        x1=f(x0) #fixpunktiterationsschritt
        error=alpha/(1-alpha)*np.abs(x1-x0) # a-posteriori
        notConverged=error>epsIncr #abbruchbedingung genauigkeit
        k=k+1
        x0=x1
    return(x1,k)

#Newtonverfahren
def newton(f, f1, x):
    return x - f(x)/f1(x)

#def newtonVerfahren(f, f1, x, tol)

def newton_easy(f, f1, x, x0): #x0 ist immer startwert
    return x - f(x)/f1(x0)

#Sekantenverfahren
def sekantenVerfahren(f, x0, x1, tol):
    count = 0
    while (abs(x0-x1) > tol and count < 10000):
        temp = x1
        x1 = x1 - ((x1 - x0)/(f(x1)-f(x0)) * f(x1))
        x0 = temp
        count += 1
    #print("sekant: " + str(x1))
    return x1

def sekant(f, x0, x1):
    return x1 - ((x1 - x0)/(f(x1)-f(x0)) * f(x1))

#Fixpunktiteration
def h(x): # Funktion
    return np.sqrt((9.81 * 3)/(np.pi * (3*2 - x)))

x = 8.995 #startwert
before = 0 #hilfsvariable für fehlerabweichung
count = 0 #iterationscounter
while (abs(x-before) > 10**-3 and count < 500):
    #print(x)
    before = x
    x = h(x)
    count += 1
print("Final: " + str(x))

# Gaussverfahren

def gauss(a):
    n = a.shape
    n = n[0]
    x = np.zeros(n)
    for i in range(n):
        if a[i][i] == 0.0:
            break
        for j in range(i + 1, n):
            ratio = a[j][i] / a[i][i]
            for k in range(n + 1):
                a[j][k] = a[j][k] - ratio * a[i][k]
    # Back Substitution
    x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = a[i][n]
        for j in range(i + 1, n):
            x[i] = x[i] - a[i][j] * x[j]
        x[i] = x[i] / a[i][i]
    print("My solution: ", x)
    a = np.delete(a, 3, axis=1)
    print("The new matrix: \n", a)
    det = 1
    for x in range(0, n):
        det = det * a[x, x]
    print("Determinante: ", det)

def GaussVerfahren(A,b):
    oDM = obereDreiecksMatrix(A, b)
    x = einsetzen(oDM)
    aODM = oDM[:,:-1]
    det = determinante(aODM)

    print("Dreiecksmatrix: ")
    print(aODM)
    print("Determinante: " + str(det))
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

#LU zerlegung
matrix2 = np.array([[0.8,2.2,3.6],[2.0,3.0,4.0],[1.2,2.0,5.8]], dtype=np.float64)
print("LU:")
print(lu(matrix2))


#QR Zerlegung
def QR(AA, b):
    A = np.copy(AA)
    Ao = np.copy(AA)
    b = np.copy(b)
    Qs = []

    cols = A.shape[0]
    for i in range(cols, 1, -1):
        print(i)
        col1 = A[:,0].reshape(i,1)
        v1 = col1 + sign(A[0][0]) * np.linalg.norm(col1,2) * e(i)
        u1 = v1/np.linalg.norm(v1,2)
        H1 = np.eye(i, dtype=float) - (np.multiply(2,np.dot(u1, u1.T)))
        Q = padding(H1, Ao)
        Qs.append(Q)
        print("H1", H1)
        print("Q",Q)
        print("Ao", Ao)
        print("Ai", np.dot(Q, AA))
        A = np.dot(Q,Ao)[1:,1:]
        print("A",A)
    print(Qs)

    #R = 
    R = np.eye(Ao.shape[0], dtype=np.float64)
    Q = np.eye(Ao.shape[0], dtype=np.float64)
    for val in Qs:
        print(val)
        print(val.T)
        Q = np.dot(Q, val.T)
        R = np.dot(R, val)
        print("Rloop", R)

    R = np.dot(R,Ao)
    print("R",R)
    print("Q",Q)
    
    print("A",np.dot(Q,R))


def sign(x):
    if x < 0:
        return -1
    return 1

def e(x):
    e = np.zeros(x, dtype=np.float64).reshape(x, 1)
    e[0][0] = 1
    return e

def padding(a, size):
    if a.shape == size.shape: return a
    result = np.zeros_like(size, dtype=float)
    xo = size.shape[0] - a.shape[0]
    result[xo:a.shape[0]+xo, xo:a.shape[1]+xo] = a
    for i in range(0, xo, 1): result[i][i] = 1
    return result


At = np.array([[1.,2,-1], [4,-2,6],[3,1,0]], dtype=np.float64)
bt = np.array([9,-4,9], dtype=np.float64).reshape(3,1)

QR(At, bt)

def QR_solve(A):
    A = np.copy(A)                       #necessary to prevent changes in the original matrix A_in
    A = A.astype('float64')              #change to float
    
    n = np.shape(A)[0]
    
    if n != np.shape(A)[1]:
        raise Exception('Matrix is not square') 
    
    Q = np.eye(n)
    R = A
    
    for j in np.arange(0,n-1):
        a = np.copy(R[j:,j:(j+1)]).reshape(n-j,1)     
        e = np.eye(n-j, dtype=float)[:,0].reshape(n-j,1)
        length_a = np.linalg.norm(a)
        if a[0] >= 0: sig = 1
        else: sig = -1
        #print("a", a)
        v = a + sig * length_a * e
        #print("v", v)
        u = v / np.linalg.norm(v)
        #print("u", u)
        H = np.eye(n-j, dtype=float) - 2 * u @ np.transpose(u)
        #print("H", H)
        Qi = np.eye(n)
        Qi[j:,j:] = H
        # print("Qi", Qi)
        R = Qi @ R
        Q = Q @ np.transpose(Qi)
        # print("R", R)
        
    return(Q,R)

A=np.array([[15, 0, 1], [1,3,7], [0, 1, 6]], dtype=np.float64)
#inverse
Ainverse = np.linalg.inv(A)
#condition infinity norm
condA = np.linalg.norm(A, np.inf) * np.linalg.norm(Ainverse, np.inf)

#Serie 9 Aufgabe 2
#relativer Fehler in gestörter Matrix
def Gruppe_6_S9_Aufg2(A, Ag, b, bg): #A, A gestört, b, b gestört
    norm = np.linalg.norm

    x = np.linalg.solve(A, b)
    xg = np.linalg.solve(Ag, bg)
    cond_A = np.linalg.cond(A, np.inf)
    
    rel_A = norm(A - Ag, np.inf) / norm(A, np.inf)
    rel_b = norm(b - bg, np.inf) / norm(b, np.inf)

    if cond_A * rel_A < 1:
        dx_max = (cond_A / (1 - (cond_A * rel_A))) * (rel_A + rel_b)
    else:
        dx_max = np.NaN
    
    dx_obs = norm(x - xg, np.inf) / norm(x, np.inf)
    
    return [x, xg, dx_max, dx_obs] #x, x gestört, obere Schranke des relativen Fehlers, tatsächlicher relatativer fehler

#LDR
A=np.array([[15, 0, 1], [1,3,7], [0, 1, 6]], dtype=np.float64)
D=np.diag(np.diag(A))
R=np.triu(A)-D
L=np.tril(A)-D

b = np.array([19,5,34], dtype=np.float64)

#Jacobi
def F_jacobi(x, b, L, D, R):
    return -np.linalg.inv(D) @ (L+R) @ x + np.linalg.inv(D) @ b

#Gauss Seidel
def F_gauss_seidel(x, b, L, D, R):
    return -np.linalg.inv(D + L) @ R @ x + np.linalg.inv(D + L) @ b

# opt: 
#   true: jacobi
#   false: gauss seidel
def Jacobi_Gauss_Seidel(A,b,x0,tol,opt): 
    D = np.diag(np.diag(A)) #diagonal matrix von A
    R = np.triu(A) - D
    L = np.tril(A) - D
    x = x0 # startwert x
    
    a_posteriori = 1
    counter = 0
    while(a_posteriori > tol):
        if(opt):
            x_next = F_jacobi(x, b, L, D, R)
        else:
            x_next = F_gauss_seidel(x, b, L, D, R)
        # print(x)
        a_posteriori = np.linalg.norm(x - x_next, np.inf)
        # print(a_posteriori)
        x = x_next
        counter += 1
    # print(counter)
    xn = x_next
    n = counter
    n2 = counter #a_posteriori?
    return [xn, n, n2] #iterationsvektor, anzahl iterationen, Anzahl benötigter Schritte gemäss der a-priori Abschätzung