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
    plt.title(r'MC history of temporal plaquette $U_t$ for $\beta=1.4$, $N_t=42$, $N_s=3$.')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.savefig(f"{path}/analysis/thermalization.png", dpi=300, bbox_inches='tight')
        
def blocksize_analysis_primary(path):
    print("reading file...")
    data = readfile(path)
    print("read file ")
    obs=data[:,25]
    boot.blocksize_analysis_primary(obs, 200, [10, 500, 10], savefig=1, path=f"{path}/analysis/")
 
def blocksize_analysis_secondary(path):
    print("reading file...")
    data = readfile(path)
    print("done.")
    
    wt = 4
    
    def potential(x):
        eps=1e-10
        return -np.log(np.clip(x, eps, None))/wt # remember 1/wt
    
    def id(x):
        return x
    
    W = data[:,1 + wt + 20]
    
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

def plot_potential_wt(path, savefig=0):
    data = readfile(path)
    
    def id(x):
        return x
    
    seed = 8220
    samples = 500
    blocksizes = [50, 50, 50]
    
    wtmax = 10
    wsmax = 3
    
    wtmaxplot = 6
    wsplot = [1, np.sqrt(5), np.sqrt(8)]
    
    plt.figure(figsize=(16,12))    
    for wt in range(wtmaxplot):
        V = []
        d_V = []
        
        for ws, blocksize in zip(range(wsmax), blocksizes):
            W = data[:, 2 + wt + wtmax*ws]

            args = [id, W]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            print(f"bootstrapping w_t={wt}, w_s={ws}")
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, samples, 1, args, seed=seed)
        
            V.append(potential(np.mean(W)))
            d_V.append(err)
        
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
        
def plot_potential_ws(path, savefig=0): ## prendere dati da file
    data = readfile(path)
    
    def id(x):
        return x
    
    seed = 8220
    samples = 500
    blocksizes = [50, 50, 50]
    
    wtmax = 10
    wsmax = 3
    
    wtmaxplot = 6
    wsplot = [1, np.sqrt(5), np.sqrt(8)]
    
    plt.figure(figsize=(16,12))    
    for ws, blocksize in zip(range(wsmax), blocksizes):
        V = []
        d_V = []
        
        for wt in range(wtmaxplot):
            W = data[:, 2 + wt + wtmax*ws]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)

            args = [id, W]
                    
            print(f"bootstrapping w_t={wt}, w_s={ws}")
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, samples, 1, args, seed=seed)
            V.append(ris) ## no
            
            d_V.append(err)
        
        plt.errorbar(1/np.arange(1,wtmaxplot+1), V, d_V, **plot.data(ws), label=fr'$w_s={wsplot[ws]:.2f}$')
        
    plt.xlabel(r'$1/w_t$')
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

def plot_fit_potential_ws(path, ws, savefig=0): ## prendere dati da file
    data = readfile(path)
    
    # for the bootstrapping of <W(wt,ws)>
    def id(x): 
        return x
    
    # for fitting aV(wt,ws) for small wt
    def ansatz(x, pars):
        return pars[0] + pars[1]/x
    
    # seed for the bootstrapping procedure to keep cross correlation intact
    seed = 8220
    
    # samples for bootstrapping
    samples = 500
    # blocksize for blocked bootstrap
    blocksize = 1000
    # initial guess of parameters
    params = [1,1]         
    
    # total number of different wt measured
    wtmax = 10
    
    # max wt used for extrapolating
    wtmaxplot = 6
        
    wsplot = [1, np.sqrt(5), np.sqrt(8)]
    
    # stuff that gets printed to file
    V_0_file = []
    d_V_0_file = []
    b_file = []
    d_b_file = []
    chi2red_file = []

    y_t0 = []
    d_y = []
                
    for wt in range(wtmaxplot):
            pb.progress_bar(wt, wtmax)
            W = data[:, 2 + wt + wtmax*ws]
            
            args = [id, W]
            
            # aV(wt,ws) = -1/wt*log(<W(wt,ws)>)
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            # computing ris and err for aV(wt,ws)
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, samples, 0, args, seed=seed, returnsamples=0)
            
            y_t0.append(potential(np.mean(W))) # on the original data
            d_y.append(err)
        
    x = np.arange(1, wtmaxplot+1)
    y_t0 = np.array(y_t0)
    d_y = np.array(d_y)           
    
    ris, err, chi2, dof, pvalue, boot_sample = reg.fit_with_yerr(x, y_t0, d_y, 3, 6, ansatz, params, samples \
                                                                    , save_figs=1, save_path=f'{path}/analysis/extrapolation/ws_{wsplot[ws]:.2f}'\
                                                                    , plot_title = fr'$aV(w_t, w_s={wsplot[ws]})$ for $\beta=1.4, N_s=3, N_t=42$', xlab = r'$w_t$', ylab = r'$aV(w_t)$')
                
    V_0_file.append(ris[0])
    d_V_0_file.append(err[0])
    b_file.append(ris[1])
    d_b_file.append(err[1])
    chi2red_file.append(chi2/dof)        
        
    data = np.column_stack((wsplot[ws], np.array(V_0_file), np.array(d_V_0_file), np.array(b_file), np.array(d_b_file), np.array(chi2red_file)))
    
    np.savetxt(f"{path}/analysis/extrapolation/ws_{wsplot[ws]:.2f}/fit_results.txt", data)

if __name__ == "__main__":
    path = "/home/negro/projects/matching/step_scaling/T42_L3_b1.4"
    
    #concatenate.concatenate(f"{path}/data", 100)
    
    #thermalization_plaqt(path)
    
    #blocksize_analysis_primary(path)
    #blocksize_analysis_secondary(path)
    
    #plot_potential_Wilsont(path, 1)
    
    #plot_potential_ws(path, 1)
    plot_fit_potential_ws(path, 0, 1)