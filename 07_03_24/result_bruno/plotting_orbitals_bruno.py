# Code Based on the SISL - SIESTA tutorial
# Make a matplotlib plot of atomic orbitals
# It assumes that you did first the calculation reported in the tutorial
# https://sisl.readthedocs.io/en/latest/tutorials/tutorial_siesta_1.html
# 
import numpy as np
from sisl import *
import matplotlib.pyplot as plt

# FUNCTION PLOT ORBITALS
def plot_atom(atom):
    no = len(atom) # number of orbitals
    nx = no // 4
    ny = no // nx
    if nx * ny < no:
        nx += 1
    fig, axs = plt.subplots(nx, ny, figsize=(20, 5*nx))
    fig.suptitle('Atom: {}'.format(atom.symbol), fontsize=14)
    def my_plot(i, orb):
            grid = orb.toGrid(atom=atom)
            # Also write to a cube file
            grid.write(r'/home/samba/homes/convidat/Desktop/07_03_24/SIESTA_examples-main/C60/result_bruno/'+'{}_{}.cube'.format(atom.symbol, orb.name()))
            c, r = i // 4, (i - 4) % 4
            if nx == 1:
                ax = axs[r]
            else:
                ax = axs[c][r]
            ax.imshow(grid.grid[:, :, grid.shape[2] // 2])
            ax.set_title(r'${}$'.format(orb.name(True)))
            ax.set_xlabel(r'$x$ [Ang]')
            ax.set_ylabel(r'$y$ [Ang]')
    i = 0
    for orb in atom:
            my_plot(i, orb)
            i += 1
    if i < nx * ny:
            # This removes the empty plots
            for j in range(i, nx * ny):
                c, r = j // 4, (j - 4) % 4
                if nx == 1:
                    ax = axs[r]
                else:
                    ax = axs[c][r]
                fig.delaxes(ax)
            plt.draw()
#
# MAIN CODE: Simply read data, print data and make a plot
#
#Read fdf file (input file of SIESTA calculation)
fdf = get_sile(r'/home/samba/homes/convidat/Desktop/07_03_24/SIESTA_examples-main/C60/result_bruno/RUN.fdf')

#Read hamiltonian obtained in SIESTA calculation
H = fdf.read_hamiltonian()
C60 = H.geometry

#Show results
print("Hamitonian:")
print(H)

plot_atom(C60.atoms[0])

plt.show()
