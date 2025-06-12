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

def readfile(path, skiprows = 1):
    output = np.loadtxt(f'{path}/analysis/dati.dat', skiprows=skiprows)
    columns = [output[:, i] for i in range(output.shape[1])]

    # plaqs plaqt polyre polyim
    
    return np.column_stack(columns)

def thermalization(path): 
    data = readfile(path)
            
    foo = data[:,2]
    
    x = np.arange(0,len(foo), len(foo)//500)
    y = foo[0:len(foo):len(foo)//500]

    plt.figure(figsize=(16,12))
    plt.plot(x, y
             , marker = 'o'
             , linestyle = '-', linewidth = 0.375
             , markersize = 2
             , color = plot.color_dict[1])
    
    plt.xlabel(r'$t_i$')
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'MC history')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.savefig(f"{path}/analysis/thermalization.png", dpi=300, bbox_inches='tight')
    
def blocksize_analysis_susc(path, h):
    
    output = np.loadtxt(f'{path}/analysis/data/dati_{h:.6f}.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    # plaqs plaqt polyre polyim
    data = np.column_stack(columns)
    
    seed = 8220
    samples_boot = 200
    
    block_vec = [5, 500, 5]
    block_range = range(block_vec[0], block_vec[1], block_vec[2])

    def id(x):
        return x

    def square(x):
        return x*x

    def susc(x):
        return x[0]-x[1]*x[1]
    
    ReP = data[:,2]
    
    list0 = [square, ReP]
    list1 = [id, ReP]

    err = []
    
    for block in block_range:
        pb.progress_bar(block, block_vec[1])
                
        _, bar = boot.bootstrap_for_secondary(susc, block, samples_boot, 0, list0, list1, seed=seed)
        
        err.append(bar)
        
    plt.figure(figsize=(16,12))
    plt.plot(block_range, err
             , marker = 'o'
             , linestyle = '-', linewidth = 0.375
             , markersize = 2
             , color = plot.color_dict[1])
        
    plt.xlabel(r'$K$')
    plt.ylabel(r'$\overline{\sigma}_{\overline{F^{(K)}}}$')
    plt.title("Standard error as a function of the blocksize.")
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)

    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.savefig(f"{path}/analysis/blocksize_analysis_susc.png", dpi=300, bbox_inches='tight')
    
def bootsusc(path, Ns):
    files = os.listdir(f'{path}/analysis/data')
    
    h_lst = []
    
    for file in files:
        tmp = file.split('_')[1].split('.dat')[0]
        h_lst.append(tmp)
        
    h_lst.sort()
    h_lst = np.array(h_lst, dtype=float)
            
    mean_lst = []
    err_lst = []
    
    for h in h_lst:
        output = np.loadtxt(f'{path}/analysis/data/dati_{h:.6f}.dat', skiprows=1)
        columns = [output[:, i] for i in range(output.shape[1])]

        # plaqs plaqt polyre polyim
        data = np.column_stack(columns)
        
        seed = 8220
        bootsamples = 500
        bootblock = 500

        def id(x):
            return x

        def square(x):
            return x*x

        def susc(x):
            return Ns**2*(x[0]-x[1]*x[1])
    
        
        ReP = np.abs(data[:,2])
        
        list0 = [square, ReP]
        list1 = [id, ReP]
                        
        mean, err = boot.bootstrap_for_secondary(susc, bootblock, bootsamples, 0, list0, list1, seed=seed)
            
        mean_lst.append(mean)
        err_lst.append(err)
    
    data = [h_lst, mean_lst, err_lst]
    np.savetxt(f"{path}/analysis/susc.txt", np.column_stack(data))

def plotsusc():
    def quadratic(x, a, b, c):
        return a + b*x +c*x**2

    plt.figure(figsize=(18,12))
    
    hmax = [6, 6, 6, 4, 4, 3]
    hmin = [3, 3, 3, 0, 0, 0]
    
    
    for i, Ns in enumerate([30, 40, 50, 60, 72, 96]):
        path = f"/home/negro/projects/reconfinement/transition/tran_10_{Ns}_23.3805/analysis"
        
        h, chi, d_chi = np.loadtxt(f"{path}/susc.txt", usecols=(0,1,2), unpack=True)
        
        plt.errorbar(h, chi, d_chi, **plot.data(i), label=fr'$N_s$={Ns}')
        
        mask = ((h<=h[hmax[i]]) & (h>=h[hmin[i]]))        
        x_mskd=h[mask]
        y_mskd=chi[mask]
        d_y_mskd=d_chi[mask]
                
        opt, cov = curve_fit(quadratic, x_mskd, y_mskd, sigma=d_y_mskd, absolute_sigma=True, p0=[1,1,1], bounds=[-np.inf,np.inf])

        x_fit = np.linspace(x_mskd[0],x_mskd[-1],500)
        y_fit = quadratic(x_fit, opt[0], opt[1], opt[2])
        
        plt.plot(x_fit, y_fit, **plot.fit(i))
        
        boot_opt = np.random.multivariate_normal(opt, cov, size=200, tol=1e-10)
        
        boot_y_fit = [quadratic(x_fit, boot_opt[j,0], boot_opt[j,1], boot_opt[j, 2]) for j in range(200)]        
        boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
        plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(i))
        
    plt.xlabel(r'$h$', size = 40)
    plt.ylabel(r'$\chi$', size = 40)
    plt.title('')
    plt.legend()
    #lt.grid(True, linestyle = '--', linewidth = 0.25)
    plt.savefig("/home/negro/projects/reconfinement/transition/chi_vs_h_vs_Ns.png", dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    Ns = 96
    
    path = f"/home/negro/projects/reconfinement/transition/tran_10_{Ns}_23.3805"

    #thermalization(path)
    #concatenate.concatenate(f"{path}/rawdata", 10000)

    #blocksize_analysis_susc(path, h=0.004200)
    
    #bootsusc(path, Ns=Ns)
    plotsusc()
