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
    output = np.loadtxt(f'{path}/data/{filename}.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    # acc_rate mag energy
    
    return np.column_stack(columns)

def get_therm(path, i, beta):
    os.makedirs(f'{path}/analysis/therm/', exist_ok=True)
    data = readfile(path, filename = f"data_{i}")
    
    mag = data[:,1]
    n_meas = len(mag)
    
    x = np.arange(0,n_meas, n_meas//500)
    y = np.abs(mag[0:n_meas:n_meas//500])
    
    np.save(f"{path}/analysis/therm/therm_mag_b{beta:.6f}", np.array([x, y], dtype=object))
  
def get_bsa(path, i, beta):
    os.makedirs(f'{path}/analysis/bsa/', exist_ok=True)
        
    data = readfile(path, filename = f"data_{i}")
    
    mag = np.abs(data[:, 1])

    x, y, d_y = boot.blocksize_analysis_primary(mag, 200, [10, len(mag)//100, 20])
    
    np.save(f"{path}/analysis/bsa/bsa_mag_b{beta:.6f}", np.array([x, y, d_y], dtype=object))
    
def get_avgmag(path, i, beta, blocksize):
    os.makedirs(f'{path}/analysis/avgmag/', exist_ok=True)
    data = readfile(path, filename = f"data_{i}")
    mag = data[:, 1]
    
    def id(x):
        return x
    
    ris, err, boot_ris = boot.bootstrap_for_primary(id, mag, blocksize, 500, 8220, True)

    np.save(f"{path}/analysis/avgmag/avgmag_b{beta:.6f}", np.array([beta, ris, err, boot_ris], dtype=object))
    
def get_avgabsmag(path, i, beta, blocksize):
    os.makedirs(f'{path}/analysis/avgabsmag/', exist_ok=True)
    data = readfile(path, filename = f"data_{i}")
    absmag = np.abs(data[:, 1])
    
    def abs(x):
        return np.abs(x)
    
    ris, err, boot_ris = boot.bootstrap_for_primary(abs, absmag, blocksize, 500, 8220, True)

    np.save(f"{path}/analysis/avgabsmag/avgabsmag_b{beta:.6f}", np.array([beta, ris, err, boot_ris], dtype=object))
    
def get_energy(path, i, beta, blocksize):
    os.makedirs(f'{path}/analysis/energy', exist_ok=True)
    data = readfile(path, filename = f'data_{i}')
    energy = data[:,2]
    
    def id(x):
        return x
    
    ris, err, boot_ris = boot.bootstrap_for_primary(id, energy, blocksize, 500, 8220, True)
    
    np.save(f"{path}/analysis/energy/energy_b{beta:.6f}", np.array([beta, ris, err, boot_ris], dtype=object))
    
def get_avgmag2(path, i, beta, blocksize):
    os.makedirs(f'{path}/analysis/avgmag2/', exist_ok=True)
    data = readfile(path, filename = f"data_{i}")
    absmag = np.abs(data[:, 1])
    
    def square(x):
        return x**2
    
    ris, err, boot_ris = boot.bootstrap_for_primary(square, absmag, blocksize, 500, 8220, True)

    np.save(f"{path}/analysis/avgmag2/avgmag2_b{beta:.6f}", np.array([beta, ris, err, boot_ris], dtype=object))
    
def get_avgmag4(path, i, beta, blocksize):
    os.makedirs(f'{path}/analysis/avgmag4/', exist_ok=True)
    data = readfile(path, filename = f"data_{i}")
    absmag = np.abs(data[:, 1])
    
    def fourth(x):
        return x**4
    
    ris, err, boot_ris = boot.bootstrap_for_primary(fourth, absmag, blocksize, 500, 8220, True)

    np.save(f"{path}/analysis/avgmag4/avgmag4_b{beta:.6f}", np.array([beta, ris, err, boot_ris], dtype=object))
 
def get_energy2(path, i, beta, blocksize):
    os.makedirs(f'{path}/analysis/energy2', exist_ok=True)
    data = readfile(path, filename = f'data_{i}')
    energy = data[:,2]
    
    def sq(x):
        return x**2
    
    ris, err, boot_ris = boot.bootstrap_for_primary(sq, energy, blocksize, 500, 8220, True)
    
    np.save(f"{path}/analysis/energy2/energy2_b{beta:.6f}", np.array([beta, ris, err, boot_ris], dtype=object))
     
    
if __name__ == "__main__":
    args = sys.argv[1:]

    L_lst = [int(val) for val in args]

    bs_lst = [2000] * len(L_lst)
    
    for L, bs in zip(L_lst, bs_lst):
        ## simulation folder
        path = f'/hiskp4/negro/Ising/FSS/L{L}'
        os.makedirs(f'{path}/analysis/', exist_ok=True)

        ## list of beta values computed
        beta_lst = [0.42 + j*0.00085 for j in range(48)]
        
        ## thermalization
        if not os.path.isdir(f"{path}/analysis/therm/"):
            for i, beta in enumerate(beta_lst):
                get_therm(path, i, beta)
        
        ## blocksize analysis
        if not os.path.isdir(f"{path}/analysis/bsa/"):
            for i, beta in enumerate(beta_lst):
                get_bsa(path, i, beta)
                
        ## average magnetization
        if not os.path.isdir(f"{path}/analysis/avgmag/"):
            for i, beta in enumerate(beta_lst):                
                get_avgmag(path, i, beta, bs)
        
        ## average absolute magnetization
        if not os.path.isdir(f"{path}/analysis/avgabsmag/"):
            for i, beta in enumerate(beta_lst):                
                get_avgabsmag(path, i, beta, bs)
                
        ## average energy
        if not os.path.isdir(f"{path}/analysis/energy/"):
            for i, beta in enumerate(beta_lst):                
                get_energy(path, i, beta, bs)
    
        ## average square magnetization
        if not os.path.isdir(f"{path}/analysis/avgmag2/"):
            for i, beta in enumerate(beta_lst):                
                get_avgmag2(path, i, beta, bs)
                
        ## average fourth power magnetization
        if not os.path.isdir(f"{path}/analysis/avgmag4/"):
            for i, beta in enumerate(beta_lst):                
                get_avgmag4(path, i, beta, bs)
                
        ## average square energy
        if not os.path.isdir(f"{path}/analysis/energy2/"):
            for i, beta in enumerate(beta_lst):                
                get_energy2(path, i, beta, bs)
    