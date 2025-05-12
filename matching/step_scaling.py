import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

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
    
def thermalization_plaqt(path): 
    data = readfile(path)
    
    plaqt = data[:,1]
    
    x = np.arange(0,len(plaqt), len(plaqt)//500)
    y = plaqt[0:len(plaqt):len(plaqt)//500]

    plt.figure(figsize=(16,12))
    plt.plot(x, y
             , marker = 'o'
             , linestyle = '-', linewidth = 0.375
             , markersize = 2
             , color = plot.color_dict[1])
    
    plt.xlabel(r'$t_i$')
    plt.ylabel(r'$U_t(t_i)$', rotation = 0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'MC history of temporal plaquette $U_t$ for $\beta=1.7$, $N_t=42$, $N_s=42$.')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.savefig(f"{path}/analysis/thermalization.png", dpi=300, bbox_inches='tight')
    
def blocksize_analysis(path):
    data = readfile(path)
    
    obs=data[:,26]
    boot.blocksize_analysis_primary(obs, 200, [10, 500, 10], savefig=1, path=f"{path}/analysis/")
    
def plot_potential_Wilsont(path, savefig=0):
    data = readfile(path)
    
    def potential(x):
        eps=1e-10
        return -np.log(np.clip(x, eps, None))
    
    def id(x):
        return x
    
    seed = 8220
    wtmax = 10
    wsmax = 3
    
    wtmaxplot = 6
    wsplot = [1, np.sqrt(5), np.sqrt(8)]
    
    blocksizes = [50, 50, 100]
    
    plt.figure(figsize=(16,12))    
    for wt in range(wtmaxplot):
        V = []
        d_V = []
        
        for ws, blocksize in zip(range(wsmax), blocksizes):
            #pb.progress_bar(ws, wsmax)
            W = data[:, 2 + wt + wtmax*ws]

            args = [id, W]
            
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, 500, 1, args, seed=seed)
            V.append(ris/(wt+1))
            d_V.append(err/(wt+1))
        
        plt.errorbar(wsplot, V, d_V, **plot.data(wt), label=fr'$w_t={wt+1}$')
        
        
    plt.xlabel(r'$w_s$')
    plt.ylabel(r'$aV(w_s)$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$ for $\beta = 1.4, N_s = 3, N_t = 42$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    if savefig==0:
        plt.show()
    elif savefig==1:
        plt.savefig(f"{path}/analysis/potential_wt.png", dpi=300, bbox_inches='tight')
        
def plot_potential_Wilsont_extrap(path, savefig=0):
    data = readfile(path)
    
    def potential(x):
        eps=1e-10
        return -np.log(np.clip(x, eps, None))
    
    def id(x):
        return x
    
    seed = 8220
    
    wtmax = 10
    wsmax = 3
    
    wtmaxplot = 6
    wsplot = [1, np.sqrt(5), np.sqrt(8)]
    
    blocksizes = [50, 50, 100]
    
    plt.figure(figsize=(16,12))    
    for ws, blocksize in zip(range(wsmax), blocksizes):
        V = []
        d_V = []
        
        for wt in range(wtmaxplot):
            W = data[:, 2 + wt + wtmax*ws]

            args = [id, W]
            
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, 500, 1, args, seed=seed)
            V.append(ris/(wt+1))
            d_V.append(err/(wt+1))
        
        plt.errorbar(1/np.arange(1,wtmaxplot+1), V, d_V, **plot.data(ws), label=fr'$w_s={wsplot[ws]:.2f}$')
        
        
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$ for $\beta = 1.4, N_s = 3, N_t = 42$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    if savefig==0:
        plt.show()
    elif savefig==1:
        plt.savefig(f"{path}/analysis/potential_ws.png", dpi=300, bbox_inches='tight')
        
def plot_fit_potential_Wilsont_extrap(path, savefig=0):
    data = readfile(path)
    
    def potential(x):
        eps=1e-10
        return -np.log(np.clip(x, eps, None))
    
    def id(x):
        return x
    
    def ansatz(x, pars):
        return pars[0] + pars[1]*x
    
    def ansatz_wrapper(x, a, b):
        pars = [a, b]
        return ansatz(x, pars)
    
    seed = 8220
    samples = 500
    
    wtmax = 10
    wsmax = 3
    
    wtmaxplot = 6
    
    wsplot = [1, np.sqrt(5), np.sqrt(8)]
    
    blocksizes = [50, 50, 100]
        
    x = range(1, wtmaxplot+1)
    
    V_0_file = []
    d_V_0_file = []
    b_file = []
    d_b_file = []
    chi2red_file = []

    plt.figure(figsize=(16,12))
    for ws, blocksize in zip(range(wsmax), blocksizes):
        y_t0 = []
        y = []
        d_y = []
        boot_y = []
            
        for wt in range(wtmaxplot):
            W = data[:, 2 + wt + wtmax*ws]
            
            args = [id, W]
            
            ris, err, bsamples = boot.bootstrap_for_secondary(potential, blocksize, samples, 1, args, seed=seed, returnsamples=1)
            
            y_t0.append(-1/(wt+1)*np.log(np.mean(W)))
            y.append(ris/(wt+1))
            d_y.append(err/(wt+1))
            boot_y.append(bsamples/(wt+1))
        
        x = np.array(x)
        y = np.array(y)
        d_y = np.array(d_y)
        boot_y = np.column_stack(boot_y)
        
        curve_fit_dict = {"p0": [1,1]}
        
        opt, cov, boot_opt, boot_cov, chi2red = reg.fit_yerr_uncorrelated(ansatz_wrapper, 1/x, y_t0, d_y, boot_y, \
        [2,5], [0,6], [0.,1.], 1, \
        curve_fit_dict, plot.data(ws), plot.fit(ws), plot.conf_band(ws), label=fr'$w_s={wsplot[ws]:.2f}$')
        
        V_0_file.append(opt[0])
        d_V_0_file.append(cov[0][0]**0.5)
        b_file.append(opt[0])
        d_b_file.append(cov[1][1]**0.5)
        chi2red_file.append(chi2red)        
        
    plt.xlabel(r'$1/w_t$')
    plt.ylabel(r'$aV(1/w_t)$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(1/w_t, w_s)$ for $\beta = 1.7, N_s = 42, N_t = 42$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    
    if savefig==0:
        plt.show()
    elif savefig==1:
        plt.savefig(f"{path}/analysis/fit_potential_ws.png", dpi=300, bbox_inches='tight')
        
    x_file = wsplot
    data = np.column_stack((x_file, np.array(V_0_file), np.array(d_V_0_file), np.array(b_file), np.array(d_b_file), np.array(chi2red_file)))

    np.savetxt(f"{path}/analysis/output.txt", data)
    
if __name__ == "__main__":
    path = "/home/negro/projects/matching/step_scaling/T42_L3_b1.4"
    
    #concatenate.concatenate(f"{path}/data", 100)
    #thermalization_plaqt(path)
    #blocksize_analysis(path)
    
    #plot_potential_Wilsont(path, 1)
    #plot_potential_Wilsont_extrap(path, 1)
    plot_fit_potential_Wilsont_extrap(path, 1)