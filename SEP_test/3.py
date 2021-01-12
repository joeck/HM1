import numpy as np
import matplotlib.pyplot as plt

#a
x = np.arange(0, np.pi, 0.01)

def F(x):
    return 1./(np.cos(x + np.pi/4) - 1) + 2

def Fd(x):
    return np.sin(x + np.pi/4)/(np.cos(x + np.pi/4) - 1)**2


plt.plot(x, F(x), label="F(x)")
plt.plot(x, Fd(x), label="Fd(x)")
plt.xlabel("x")
plt.xlim(0, np.pi)
plt.ylim(0, np.pi)
plt.legend()

#b
xL=1
xR=2

fx = F(np.array([xL, xR]))
fxMin = np.min(fx)
fxMax = np.max(fx)
fdx = Fd(np.array([xL, xR]))
alpha = np.max(fdx)

print("Banachschen Fixpunktsatz:")
print(xL<fxMin and fxMax < xR and alpha < 1 and 0<alpha) #true
print("Lipschitzkonstante: " + str(alpha))

#c
print("Schätzwert:")
xj = 1.3376
xj1 = 1.3441

estX = alpha/(1-alpha) * np.abs(xj1 - xj)
print(estX)
# Schätzwert:
# 0.012850688283082362

#d
print("Fixpunktiteration d)")
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
x0=1
[xF,n]=fixpunktIteraion(F,x0,1e-6,alpha)
print('xF=%.4E'% xF,'(n=%.4E Iterationen)\n' %n)
#xF=1.3478E+00 (n=1.5000E+01 Iterationen)

#e)
def fA(x): return (x-1)/(x-2) - np.cos(x + np.pi/4)
def fB(x): return F(x)
def fC(x): return x + (2 - 1/(np.cos(x+np.pi/4) - 1))
def fD(x): return np.cos(x + np.pi/4) - 1/(x-2) - 1

print("A ist ", np.abs(fA(xF))<1e-6)
print("B ist ",np.abs(fB(xF))<1e-6)
print("C ist ",np.abs(fC(xF))<1e-6)
print("D ist ",np.abs(fD(xF))<1e-6)

print(fA(xF))
print(fB(xF))
print(fC(xF))
print(fD(xF))