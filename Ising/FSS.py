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

def thermalization(path, i, beta):
    data = readfile(path, filename = f"data_{i}")
    
    mag = data[:,1]
    n_meas = len(mag)
    
    x = np.arange(0,n_meas, n_meas//500)
    y = np.abs(mag[0:n_meas:n_meas//500])
    
    plt.figure(figsize=(18,12))
    plt.plot(x, y
                    , marker = 'o'
                    , linestyle = '-', linewidth = 0.375
                    , markersize = 2
                    , color = plot.color_dict[1])
    
    plt.xlabel(r'$t_i$')

    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"{path}/analysis/therm/therm_b{beta}.png", bbox_inches='tight')
    plt.close()    

def bsa_observable(path, i, beta):
    os.makedirs(f'{path}/analysis/bsa/', exist_ok=True)
        
    data = readfile(path, filename = f"data_{i}")
    
    mag = np.abs(data[:, 1])

    boot.blocksize_analysis_primary(mag, 200, [10, len(mag)//100, 20], savefig=1, path=f"{path}/analysis/bsa", extra=f'_b{beta:.6f}')
        
def get_mag(path, i, blocksize):
    data = readfile(path, filename = f"data_{i}")
    mag = np.abs(data[:, 1])
    
    def abs(x):
        return np.abs(x)
    
    ris, err, boot_ris = boot.bootstrap_for_primary(abs, mag, blocksize, 200, 8220, True)
    
    return ris, err, boot_ris
    
if __name__ == "__main__":
    L = 40
    blocksize = 2000
    path = f'/home/negro/projects/misc/Ising/L{L}'
    
    beta_lst = [0.042 + j*0.00085 for j in range(48)]
    if not os.path.isdir(f"{path}/analysis/therm/"):
        for i, beta in enumerate(beta_lst):
            thermalization(path, i, beta)
            
    if not os.path.isdir(f"{path}/analysis/bsa/"):
        for i, beta in enumerate(beta_lst):
            bsa_observable(path, i, beta)
    
    ris_lst, err_lst, boot_lst = [], [], []
    if not os.path.isfile(f"{path}/analysis/mag.npy"):
        for i, beta in enumerate(beta_lst):
            pb.progress_bar(i, len(beta_lst), 50, 0)
            
            ris, err, boot_ris = get_mag(path, i, blocksize)
            ris_lst.append(ris)
            err_lst.append(err)
            boot_lst.append(boot_ris)
        
        np.save(f"{path}/analysis/mag", np.array([beta_lst, ris_lst, err_lst, boot_lst], dtype=object))
    
    plt.figure()
    for i, L in enumerate([20]):
        path = f'/home/negro/projects/misc/Ising/L{L}'
        data = np.load(f"{path}/analysis/mag.npy", allow_pickle=True)
        x = data[0]
        y = data[1]
        d_y = data[2]
        
        plt.errorbar(x, y, d_y, **plot.data(i))
    plt.show()
    
    
    