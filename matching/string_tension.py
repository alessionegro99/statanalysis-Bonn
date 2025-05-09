import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

import plot
import progressbar as pb
import bootstrap as boot
import regression as reg

def readfile(path):
    output = np.loadtxt(f'{path}/analysis/dati.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    # plaqs plaqt ReP ImP W(wt=1,ws=1) W(Wt=1,ws=2) ... W(wt=1,ws=10) W(wt=2,ws=10) ... W(wt=10,ws=10)
    
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
    
    obs=data[:,1]
    boot.blocksize_analysis_primary(obs, 200, [2, 500, 10], savefig=1, path=f"{path}/analysis/")
    
def plot_potential_Wilsont(path, savefig=0):
    data = readfile(path)
    
    def potential(x):
        eps=1e-10
        return -np.log(np.clip(x, eps, None))
    
    def id(x):
        return x
    
    seed = 8220
    wtmax = 10
    wsmax = 10
    
    wtmaxplot = 5
    wsplot = range(1, wsmax+1)
    
    plt.figure(figsize=(16,12))
        
    for wt in range(wtmaxplot):
        V = []
        d_V = []
        
        for ws in range(wsmax):
            pb.progress_bar(ws, wsmax)
            W = data[:, 4 + ws + wsmax*wt]

            args = [id, W]
            
            ris, err = boot.bootstrap_for_secondary(potential, 50, 500, 0, args, seed=seed)
            V.append(ris/(wt+1))
            d_V.append(err/(wt+1))
        
        plt.errorbar(wsplot, V, d_V, **plot.data(wt), label=fr'$w_t={wt+1}$')
        
        
    plt.xlabel(r'$w_s$')
    plt.ylabel(r'$aV(w_s)$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$ for $\beta = 1.7, N_s = 42, N_t = 42$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    if savefig==0:
        plt.show()
    elif savefig==1:
        plt.savefig(f"{path}/analysis/potential_wt.png", dpi=300, bbox_inches='tight')
  
def plot_potential_ws(path, savefig=0):
    data = readfile(path)
    
    def potential(x):
        eps=1e-10
        return -np.log(np.clip(x, eps, None))
    
    def id(x):
        return x
    
    seed = 8220
    wtmax = 10
    wsmax = 10
    
    wtmaxplot = 5
    wtplot = np.arange(1, wtmaxplot+1)
    
    plt.figure(figsize=(16,12))
    
    for ws in range(wsmax):
        V = []
        d_V = []
        
        for wt in range(wtmaxplot):
            pb.progress_bar(wt, wtmax)
            W = data[:, 4 + ws + wtmax*wt]
            
            args = [id, W]
            
            ris, err = boot.bootstrap_for_secondary(potential, 50, 500, 0, args, seed=seed)
            V.append(ris/(wt+1))
            d_V.append(err/(wt+1))
        
        plt.errorbar(1/wtplot, V, d_V, **plot.data(ws), label=fr'$w_t={ws+1}$')

    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$ for $\beta = 1.7, N_s = 42, N_t = 42$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    
    if savefig==0:
        plt.show()
    elif savefig==1:
        plt.savefig(f"{path}/analysis/potential_ws.png", dpi=300, bbox_inches='tight')
        
def plot_fit_potential_ws(path, savefig=0):
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
    samples = 200
    
    wtmax = 10
    wsmax = 10
    
    wtmaxplot = 5
    wtplot = 1/np.arange(1, wtmaxplot+1)
        
    x = range(1, wtmaxplot+1)

    plt.figure(figsize=(16,12))
    for ws in range(wsmax):
        y_t0 = []
        y = []
        d_y = []
        boot_y = []
            
        for wt in range(wtmaxplot):
            pb.progress_bar(wt, wtmax)
            W = data[:, 4 + ws + wtmax*wt]
            
            args = [id, W]
            
            ris, err, bsamples = boot.bootstrap_for_secondary(potential, 50, samples, 0, args, seed=seed)
            
            y_t0.append(-1/(wt+1)*np.log(np.mean(W)))
            y.append(ris/(wt+1))
            d_y.append(err/(wt+1))
            boot_y.append(bsamples/(wt+1))
        
        x = np.array(x)
        y = np.array(y)
        d_y = np.array(d_y)
        boot_y = np.column_stack(boot_y)
        
        curve_fit_dict = {"p0": [1,1]}
        opt, cov, boot_opt, boot_cov = reg.fit_yerr_uncorrelated(ansatz_wrapper, 1/x, y_t0, d_y, boot_y, \
        [2,5], [0,5], [0.,1.], 1, \
        curve_fit_dict, plot.data(ws), plot.fit(ws), plot.conf_band(ws))
        
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$ for $\beta = 1.7, N_s = 42, N_t = 42$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    
    if savefig==0:
        plt.show()
    elif savefig==1:
        plt.savefig(f"{path}/analysis/potential_ws.png", dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    path = "/home/negro/projects/matching/string_tension/L42_b1.7"
    
    #plot_potential_Wilsont(path, savefig=1)
    #plot_potential_ws(path, savefig=1)
    #plot_fit_potential_ws(path, savefig=0)
    
    plot_fit_potential_ws(path)