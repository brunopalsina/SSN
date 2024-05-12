import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.rdf as rdf
import MDAnalysis.analysis.align as align
import warnings
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def Figure3D(path, title, header1, header2, header3, num=1):
    data = pd.read_csv(path)
    x = data[header1].values
    y = data[header2].values
    z = data[header3].values
    aspect_ratio = 1.1
    fig_height = 8
    fig_width = fig_height * aspect_ratio
    if num == None:
        fig = plt.figure()
    else: 
        fig = plt.figure(num, figsize=(fig_width, fig_height))
        fig.set_facecolor('lightgrey')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#D3D3D3')
        
    # Plot the data points
    #ax.scatter(x, y, z, c='r', marker='o')
    ax.plot_trisurf(x, y, z, cmap='plasma', linewidth=0.2) #color maps: coolwarm, viridis, plasma, magma, jet

    # Set labels and title
    ax.set_xlabel(header1)
    ax.set_ylabel(header2)
    ax.set_zlabel(header3)
    ax.set_title(title, y=1.04)
    fig.tight_layout()
    plt.show()

Figure3D('/home/bruno/Uni/SSN/MDAnalysis/data.csv', 'RDF as a function of T','Distance', 'Temperature', 'RDF')