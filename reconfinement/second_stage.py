import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.special import k0
from scipy.linalg import eig

import concatenate
import plot

import progressbar as pb
import bootstrap as boot
import regression as reg

def plot_res_firstage(path, beta, h):
    fit_res = np.loadtxt(f'{path}/b{beta}_h{h}/results_b{beta}_h{h}.txt', skiprows=1)
    
    x = fit_res[:,0]
    y = fit_res[:,5]
    d_y = fit_res[:,6]

    plt.figure(figsize=(18,12))
    
    plt.errorbar(x, y, d_y, **plot.data(1))
    plt.savefig(f'{path}/b{beta}_h{h}/results_b{beta}_h{h}.png', bbox_inches='tight', dpi=300)
    plt.close()    

if __name__ == '__main__':
    path = '/home/negro/projects/reconfinement/polycorr_Nt'
    
    beta = 23.3805
    h = 0.005
    
    plot_res_firstage(path, beta, h)

