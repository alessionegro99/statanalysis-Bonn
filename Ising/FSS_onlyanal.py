import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import fsolve

import concatenate
import plot
import progressbar as pb
import bootstrap as boot
import regression as reg

def readfile(path, filename):
    output = np.loadtxt(f'{path}/{filename}', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]
    
    return np.column_stack(columns)

def plot_bsa(path, Ns, beta):
    x, y, d_y = np.load(f'{path}/L{Ns}/analysis/therm/bsa_mag_b{beta:.6f}.npy', allow_pickle=True)
    
    plt.figure(figsize=(18,12))
    
    plt.errorbar(x, y, d_y,
                  fmt='o-', capsize=3, 
                  markersize=2, linewidth=0.375,
                  color=plot.color_dict[1])    
    plt.xlabel(r'$K$')
    plt.ylabel(r'$\overline{\sigma}_{\overline{F^{(K)}}}$', rotation=0)
    plt.title("Standard error as a function of the blocksize.")

    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.savefig(f'{path}/L{Ns}/analysis/bsa/bsa_mag_b{beta:.6f}.png', bbox_inches='tight')
    
def plot_avgabsmag(path, Ns_lst, beta_lst):
    
    def m0beta(x):
        x = np.asarray(x)
        threshold = 0.5 * np.log(1 + 2**0.5)
        result = np.zeros_like(x)
        
        valid_mask = x >= threshold
        result[valid_mask] = (1 - np.sinh(2 * x[valid_mask])**(-4))**(1/8)
        
        return result

    plt.figure(figsize=(18,12))
    for i, Ns in enumerate(Ns_lst):
        data = [
            np.load(f'{path}/L{Ns}/analysis/mag/mag_b{beta:.6f}.npy', allow_pickle=True)
            for beta in beta_lst
        ]

        x, y, d_y, b_y = map(list, zip(*data))
        plt.errorbar([beta for beta in beta_lst],y,d_y, **plot.data(i), label=f'L={Ns}')
    
    x_linsp = np.linspace(beta_lst[0], beta_lst[-1], 100)
    y_linsp = m0beta(x_linsp)
    
    plt.plot(x_linsp, y_linsp, **plot.fit(2))

    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\langle |m| \rangle$')
    plt.legend()
    plt.savefig(f"{path}/avgabsmag.png", bbox_inches='tight')

def plot_energy(path, Ns_lst, beta_lst):
    plt.figure(figsize=(18,12))
    
    for i, Ns in enumerate(Ns_lst):
        data = [
            np.load(f'{path}/L{Ns}/analysis/mag/mag_b{beta:.6f}.npy', allow_pickle=True)
            for beta in beta_lst
        ]

        x, y, d_y, b_y = map(list, zip(*data))
        plt.errorbar([beta for beta in beta_lst],y,d_y, **plot.data(i), label=f'L={Ns}')
    
    x_linsp = np.linspace(beta_lst[0], beta_lst[-1], 100)
    y_linsp = m0beta(x_linsp)
    
    plt.plot(x_linsp, y_linsp, **plot.fit(2))

    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\langle |m| \rangle$')
    plt.legend()
    plt.savefig(f"{path}/avgabsmag.png", bbox_inches='tight')
    
if __name__ == "__main__":
    Ns_lst = [40, 60, 80, 100, 120, 140, 160]
    
    beta_lst = [0.42 + j*0.00085 for j in range(48)]

    path = f'/home/negro/projects/misc/Ising'
    
    #plot_bsa(path, 160, 0.079400)
    
    plot_avgabsmag(path, Ns_lst, beta_lst)


