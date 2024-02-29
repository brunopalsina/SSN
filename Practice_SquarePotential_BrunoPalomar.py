# ----------------------------
# Quantum Potential Well
# ATOMIC UNITS
# V=4 for x<-2 or x>2
# ----------------------------
# Finite differences method as developed by Truhlar JCP 10 (1972) 123-132
#
# code by Jordi Faraudo
#
#
import numpy as np
import matplotlib.pyplot as plt

#Potential as a function of position
def getV(x):
    if (abs(x)<np.pi):
       potvalue = 5*(np.cos(x))**2
    else:
       potvalue = 5
    return potvalue

#Discretized Schrodinger equation in n points (FROM 0 to n-1)
def Eq(n,h,x):
    F = np.zeros([n,n])
    for i in range(0,n):
        F[i,i] = -2*((h**2)*getV(x[i]) + 1)
        if i > 0:
           F[i,i-1] = 1
           if i < n-1:
              F[i,i+1] = 1
    return F

#-------------------------
# Main program
#-------------------------
# Interval for calculating the wave function [-L/2,L/2]
L = 2*np.pi
xlower = -L/2.0
xupper = L/2.0

#Discretization options
h = 0.002  #discretization in space

#Create coordinates at which the solution will be calculated
x = np.arange(xlower,xupper+h,h)
#grid size (how many discrete points to use in the range [-L/2,L/2])
npoints=len(x)

print("Using",npoints, "grid points.")

#Calculation of discrete form of Schrodinger Equation
print("Calculating matrix...")
F=Eq(npoints,h,x)

#diagonalize the matrix F
print("Diagonalizing...")
eigenValues, eigenVectors = np.linalg.eig(F)

#Order results by eigenvalue
# w ordered eigenvalues and vs ordered eigenvectors
idx = eigenValues.argsort()[::-1]
w = eigenValues[idx]
vs = eigenVectors[:,idx]**2

#Energy Level
E = - w/(2.0*h**2)

#Print Energy Values
print("RESULTS:")
for k in range(0,4):
  Erel=E[k]/2.0  #ratio between energy level and potential well
  print("State ",k,": Energy = %.4f" %E[k],', E/V='+'{:.4f}'.format(Erel))

#Init Wavefunction (empty list with npoints elements)
psi = [None]*npoints

#Calculation of normalised Wave Functions
for k in range(0,len(w)):
	psi[k] = vs[:,k]
	integral = h*np.dot(psi[k],psi[k])
	psi[k] = psi[k]/integral**0.5

#Plot Wave functions
print("Plotting")

#v = int(input("\n Quantum Number (enter 0 for ground state):\n>"))
v = 0
while E[v] <= getV(L+1):
    fig = plt.figure(v+1)
    ax = fig.subplots()
    ax2 = ax.twinx()
    ax.plot(x,psi[v],label=r'$\psi_v(x)²$, k = ' + str(v))
    ax2.plot(x, list(map(getV, x)), color = 'orange',label = 'V(x)')
    ax.set_title(r'$n=$'+ str(v) + r', $E_n$=' + '{:.4f}'.format(E[v]))
    ax.legend()
    ax.set_xlabel(r'$x$ (a.u.)')
    ax.set_ylabel(r'$\psi(x)²$')
    ax2.set_ylabel('V(x)')
    v += 1
    plt.show()

print("Bye")
