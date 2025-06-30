import sys
import os
import shutil

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

def readfile(path):
    output = np.loadtxt(f'{path}/analysis/dati.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    # plaqs plaqt W(wt=1,ws=1) ... W(wt=10,ws=1) W(wt=1,ws=sqrt(5)) ... W(wt=10,ws=sqrt(5)) W(wt=10,ws=1) W(wt=1,ws=sqrt(8)) ... W(wt=10,ws=sqrt(8))
    
    return np.column_stack(columns)
    
def thermalization(path): 
    output = np.loadtxt(f'{path}/data/dati_0.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    # plaqs plaqt W(wt=1,ws=1) ... W(wt=wtmax,ws=1) W(wt=1,ws=sqrt(5)) ... W(wt=10,ws=sqrt(5)) W(wt=10,ws=1) W(wt=1,ws=sqrt(8)) ... W(wt=10,ws=sqrt(8))
    
    data = np.column_stack(columns)
    
    foo = data[:,2 + 24 + 24 + 10]
    
    x = np.arange(0,len(foo), len(foo)//500)
    y = foo[0:len(foo):len(foo)//500]
    
    np.savetxt(f"{path}/analysis/thermalization.txt", [x,y])

def blocksize_analysis_primary(path):
    data = readfile(path)
    
    obs=data[:,2]
    boot.blocksize_analysis_primary(obs, 100, [10, 500, 5], savefig=0, path=f"{path}/analysis/", extra='r1')
 
    obs=data[:,2 + 24]
    boot.blocksize_analysis_primary(obs, 100, [10, 500, 5], savefig=0, path=f"{path}/analysis/", extra='r2')
 
    obs=data[:,2 + 48]
    boot.blocksize_analysis_primary(obs, 100, [10, 500, 5], savefig=0, path=f"{path}/analysis/", extra='r3')
 
def blocksize_analysis_secondary(path):
    data = readfile(path)
    
    wt = 4
    
    def potential(x):
        eps=1e-10
        return -np.log(np.clip(x, eps, None))/wt # remember 1/wt
    
    def id(x):
        return x
    
    W = data[:,1 + wt + 10]
    
    seed = 8220
    args = [id, W]

    block_vec = [10, 500, 10]

    block_range = range(block_vec[0], block_vec[1], block_vec[2])

    err = []
    
    samples_boot = 200
    for block in block_range:
        pb.progress_bar(block, block_vec[1])
                
        foo, bar = boot.bootstrap_for_secondary(potential, block, samples_boot, 0, args, seed=seed)
        
        err.append(bar)
        
    plt.figure(figsize=(16,12))
    plt.plot(block_range, err
             , marker = 'o'
             , linestyle = '-', linewidth = 0.375
             , markersize = 2
             , color = plot.color_dict[1])
        
    plt.xlabel(r'$K$')
    plt.ylabel(r'$\overline{\sigma}_{\overline{F^{(K)}}}$', rotation=0)
    plt.title("Standard error as a function of the blocksize.")
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)

    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.savefig(f"{path}/analysis/blocksize_analysis_secondary_W(4,sqrt(5)).png", dpi=300, bbox_inches='tight')

def get_potential_wt(path, wsplot, wtmax):
    data = readfile(path)
    def id(x):
        return x
    
    seed = 8220
    samples = 500
    blocksizes = [50, 50, 50]
    
    wsmax = 3
    
    wtmaxplot = wtmax
    
    wsplot = np.array(wsplot)
    
    V = []
    d_V = []
    V_bs = []
    
    for wt in range(wtmaxplot):
        V_wt = []
        d_V_wt = []
        V_bs_wt = []
        
        for ws, blocksize in zip(range(wsmax), blocksizes):
            W = data[:, 2 + wt + wtmax*ws]

            args = [id, W]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            _, err, bs = boot.bootstrap_for_secondary(potential, blocksize, samples, 1, args, seed=seed, returnsamples=1)
        
            sys.stdout.flush()

            V_wt.append(potential(np.mean(W)))
            d_V_wt.append(err)
            V_bs_wt.append(bs)
        
        plt.errorbar(wsplot, V_wt, d_V_wt, **plot.data(wt), label=fr'$w_t={wt+1}$')
        
        V.append(V_wt)
        d_V.append(d_V_wt)
        V_bs.append(V_bs_wt)
        
    np.save(f"{path}/analysis/potential_wt", np.array([wsplot, V, d_V, V_bs], dtype=object))
            
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <Ns> <beta1> <beta2> ...")
        sys.exit(1)
    
    Ns = int(sys.argv[1])
    betas = [float(b) for b in sys.argv[2:]]
    betas = [('{:g}'.format(b)) for b in betas]

    for beta in betas:
        path_glob = f"/lustre/scratch/data/anegro_hpc-matching/step_scaling/L{Ns}/T48_L{Ns}_b{beta}"
        
        if not os.path.exists(f"{path_glob}/analysis"):
            os.makedirs(f"{path_glob}/analysis")
        
        #thermalization(path_glob)

        #concatenate.concatenate(f"{path_glob}/data", 10000, f"{path_glob}/analysis")
            
        blocksize_analysis_primary(path)
        #blocksize_analysis_secondary(path)
       
        if Ns==3:
            wsplot = [1, np.sqrt(5), np.sqrt(8)]
        elif Ns==4:
            wsplot = [np.sqrt(2), np.sqrt(10), np.sqrt(18)]
        elif Ns==5:
            wsplot = [np.sqrt(5), np.sqrt(25), np.sqrt(32)]
        elif Ns==7:
            wsplot = [np.sqrt(8), np.sqrt(40), np.sqrt(72)]
        
        wtmax = 24

       # get_potential_wt(path_glob, wsplot,wtmax=wtmax)
