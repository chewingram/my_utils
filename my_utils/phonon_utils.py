import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import matplotlib

def ph_plot(xx, yy, nmodes, plot_range, unit_conv=1, xaxis_lab='x', yaxis_lab='y', xticks=None, xticks_lab=None, zero=True, vlines=True, vlines_x=None, save=False, fname='phonon_fig', **kwargs):
    '''
    Function to plot phonons having the x and y data
    Parameters:
    xx (nunmpy array): x axis (common for all modes)
    yy (list, numpy array): list of numpy arrays, one for each mode
    nmodes (int): number of modes to print
    plot_range (2-tuple): 2-tuple with the plot range (in x-data units)
    unit_conv (float): conversion unit for the y data
    xaxis_lab (str): label for x axis
    yaxis_lab (str): label for y axis
    xticks (list, float): list of xticks (in data units)
    xticks_lab (list, str): list of labels for the xticks
    zero (bool): horizontal line at y=0
    save (bool): save image as png with dpi=600
    fname (str): name to use to save the figure (if 'save' is True)
    kwargs: everything is given to the pyplot.plot() function
    '''
    if vlines == True and vlines_x == None:
        raise TypeError("Since 'vlines' is True, please give the positions of the vertical lines.")
        
    plot_lines = []
    
    fig1 = plt.figure()
    ax = fig1.add_axes((0, 0, 1, 1))
    for i in range(nmodes):
        plot_lines.append(ax.plot(xx, unit_conv*yy[i], color='black', **kwargs))
    ax.set_xticks(xticks, xticks_lab)
    ax.set_ylabel(yaxis_lab)
    ax.set_xlabel(xaxis_lab)

    # Drawing the vertical lines
    if vlines == True:
        vlines = []
        for x in vlines_x:
            vlines.append(plt.axvline(x, color="black", linewidth=1, alpha=0.5, zorder=-1))

    if zero == True:
        # Draw the zero energy line
        plt.axhline(0, color="black", linewidth=1, alpha=0.5, zorder=-1)

    plt.xlim((plot_range[0], plot_range[1]))
    if save == True:
        plt.savefig(fname + ".png", format='png', bbox_inches='tight', dpi=600)