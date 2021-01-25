import numpy as np
import matplotlib.pyplot as plt

#a
def f(x):
    return 1/(np.cos(x + np.pi/4) - 1) + 2

def df(x):
    return np.sin(x + np.pi/4)/((np.cos(x + np.pi/4) - 1)**2)

x = np.arange(0, np.pi, 0.01)

plt.plot(x,f(x), label="f(x)")
plt.plot(x, df(x), label="df(x)")

#b
xL = 1
xR = 2
fx = f(np.array([xL, xR]))
fxmin = np.min(fx)
fxmax = np.max(fx)
a = np.max(df(np.arange(xL, xR, 0.01)))
print(a) #Lipschitz: 0.6640946355544965
print("Banachsche: ")
print(xL < fxmin and fxmax < xR and 0 < a and a < 1 ) #True

#c
x1 = 1.3376
x2 = 1.3441
aPost = a/(1-a) * np.abs(x2 - x1)
print(aPost) #0.012850688283082362

#d
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
    return(x1,k) # berechneter Fixpunkt, anzahl iterationen

[xF, n] = fixpunktIteraion(f, 1, 1e-6, a)
print(xF)

#e
def fA(x): return (x-1)/(x-2) - np.cos(x + np.pi/4)
def fB(x): return f(x)
def fC(x): return x + (2 - 1/(np.cos(x+np.pi/4) - 1))
def fD(x): return np.cos(x + np.pi/4) - 1/(x-2) - 1

print("A ist ", np.abs(fA(xF))<1e-6)
print("B ist ",np.abs(fB(xF))<1e-6)
print("C ist ",np.abs(fC(xF))<1e-6)
print("D ist ",np.abs(fD(xF))<1e-6)