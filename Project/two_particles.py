import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def eVtoJoule(volt):
    return volt*10e-3/6,242e+18

def eVtoKelvin(volt):
    return volt*10e-3*11606

global sigma, epsilon
sigma = 0.34 #nm
epsilon = 2.844 #meV
xmin = (2**(1/6))*sigma*1e-9
x = np.linspace(0.315, 1, 51)

def lj_pot(x, epsilon, sigma):
    pot = 4.0 * epsilon * ((sigma / x)**12 - (sigma / x)**6)
    return pot

def force(x):
    der = -(24*epsilon*(sigma**6)*(x**6 - 2*sigma**6))/(x**13)
    return der

mapeada = map(lj_pot, x, [epsilon for i in range(len(x))], [sigma for i in range(len(x))])
lj_pot_list = list(mapeada)

fig = plt.figure(1, (12,9))
ax = fig.subplots()

ax.plot(x, lj_pot_list)
ax.plot(x, force(x)/50, color='cyan')

plt.show()
