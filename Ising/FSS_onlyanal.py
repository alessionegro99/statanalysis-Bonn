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
    x, y, d_y = np.load(f'{path}/L{Ns}/therm/bsa_mag_b{beta:.6f}.npy', allow_pickle=True)
    
    plt.figure(figsize=(18,12))
    
    plt.errorbar(x, y, d_y,
                  fmt='o-', capsize=3, 
                  markersize=2, linewidth=0.375,
                  color=plot.color_dict[1])    
    plt.xlabel(r'$K$')
    plt.ylabel(r'$\overline{\sigma}_{\overline{F^{(K)}}}$', rotation=0)
    plt.title("Standard error as a function of the blocksize.")

    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.savefig(f'{path}/L{Ns}/bsa/bsa_mag_b{beta:.6f}.png', bbox_inches='tight', dpi=300)
    
def plot_avgmag(path, Ns_lst, beta_lst):
    plt.figure(figsize=(18,12))
    for i, Ns in enumerate(Ns_lst):
        data = [
            np.load(f'{path}/L{Ns}/avgmag/avgmag_b{beta:.6f}.npy', allow_pickle=True)
            for beta in beta_lst
        ]

        x, y, d_y, b_y = map(list, zip(*data))
        plt.errorbar(x,y,d_y, **plot.data(i), label=f'L={Ns}')
        
    plt.ylim(-0.025, 0.025)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\langle m \rangle$')
    plt.legend()
    plt.savefig(f"{path}/avg_mag.png", bbox_inches='tight', dpi=300)
    
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
            np.load(f'{path}/L{Ns}/avgabsmag/avgabsmag_b{beta:.6f}.npy', allow_pickle=True)
            for beta in beta_lst
        ]

        x, y, d_y, b_y = map(list, zip(*data))
        plt.errorbar(x,y,d_y, **plot.data(i), label=f'L={Ns}')
    
    x_linsp = np.linspace(beta_lst[0], beta_lst[-1], 100)
    y_linsp = m0beta(x_linsp)
    
    plt.plot(x_linsp, y_linsp, **plot.fit(2))

    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\langle |m| \rangle$')
    plt.legend()
    plt.savefig(f"{path}/avgabsmag.png", bbox_inches='tight', dpi=300)

def plot_energy(path, Ns_lst, beta_lst):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    plt.figure(figsize=(18,12))
    ax = plt.gca()
    
    inset = inset_axes(ax, width="30%", height="30%", loc='upper right')  # Inset axes
    inset.set_xlim(0.4365, 0.4385)
    inset.set_ylim(-1.43, -1.36)
    
    for i, Ns in enumerate(Ns_lst):
        data = [
            np.load(f'{path}/L{Ns}/energy/energy_b{beta:.6f}.npy', allow_pickle=True)
            for beta in beta_lst
        ]

        x, y, d_y, b_y = map(list, zip(*data))
        ax.errorbar(x,y,d_y, **plot.data(i), label=f'L={Ns}')
        inset.errorbar(x, y, d_y, **plot.data(i))
        
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\epsilon$')
    ax.legend(loc='lower left')
    plt.savefig(f"{path}/energy.png", bbox_inches='tight', dpi=300)
    plt.close()

def get_chi(path, Ns, beta):
    x_m, m, d_m, b_m = np.load(f'{path}/L{Ns}/avgabsmag/avgabsmag_b{beta:.6f}.npy', allow_pickle=True)
    x_m2, m2, d_m2, b_m2 = np.load(f'{path}/L{Ns}/avgmag2/avgmag2_b{beta:.6f}.npy', allow_pickle=True)

    chi, d_chi = Ns**2*m2, Ns**2*d_m2
    chip = Ns**2*(m2-m**2)
    
    b_chip = []
    for b_m_j, b_m2_j in zip(b_m, b_m2):
        b_chip.append((b_m2_j-b_m_j**2))
    
    d_chip = np.std(b_chip, ddof=1)
    
    return x_m, chi, d_chi, chip, d_chip

def plot_chi(path, Ns_lst, beta_lst):
    plt.figure(figsize = (18,12))
    for i, Ns in enumerate(Ns_lst):
        for beta in beta_lst:
            x, chi, d_chi, _, _ = get_chi(path, Ns, beta)
            plt.errorbar(x, chi/1e3, d_chi/1e3, **plot.data(i))
    
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\chi/10^3$')
    plt.savefig(f"{path}/chi.png", bbox_inches='tight', dpi=300)
    plt.close()
    
def plot_chip(path, Ns_lst, beta_lst):
    plt.figure(figsize = (18,12))
    for i, Ns in enumerate(Ns_lst):
        for beta in beta_lst:
            x, _, _, chip, d_chip = get_chi(path, Ns, beta)
            plt.errorbar(x, chip/100, d_chip/100, **plot.data(i))
    plt.savefig(f"{path}/chip.png", bbox_inches='tight', dpi=300)
    plt.close()    
    
def get_binder(path, Ns, beta):
    x_m2, m2, d_m2, b_m2 = np.load(f'{path}/L{Ns}/avgmag2/avgmag2_b{beta:.6f}.npy', allow_pickle=True)
    x_m4, m4, d_m4, b_m4 = np.load(f'{path}/L{Ns}/avgmag4/avgmag4_b{beta:.6f}.npy', allow_pickle=True)

    binder = m4/m2**2

    b_m2=np.sum(b_m2, axis=1)/len(b_m2)
    b_m4=np.sum(b_m4, axis=1)/len(b_m4)

    
    b_binder = []
    for b_m2_j, b_m4_j in zip(b_m2, b_m4):
        b_binder.append((b_m4_j/b_m2_j**2))
    
    d_binder = np.std(b_binder, ddof=1)
    
    return x_m2, binder, d_binder
    
def plot_binder(path, Ns_lst, beta_lst):
    plt.figure(figsize = (18,12))
    for i, Ns in enumerate(Ns_lst):
        for beta in beta_lst:
            x, y, d_y = get_binder(path, Ns, beta)
            plt.errorbar(x, y, d_y, **plot.data(i))
    plt.savefig(f"{path}/binder.png", bbox_inches='tight', dpi=300)
    plt.close()     
    
if __name__ == "__main__":
    Ns_lst = [20, 40, 60, 80, 100, 120, 140, 160]
    
    beta_lst = [0.42 + j*0.00085 for j in range(48)]

    path = f'/home/negro/projects/misc/Ising'
    
    #plot_avgmag(path, Ns_lst, beta_lst)
    #plot_avgabsmag(path, Ns_lst, beta_lst)
    #plot_energy(path, Ns_lst, beta_lst)
    
    #plot_chi(path, Ns_lst, beta_lst)
    plot_chip(path, Ns_lst, beta_lst)
    plot_binder(path, Ns_lst, beta_lst)
