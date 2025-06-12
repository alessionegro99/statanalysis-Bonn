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
    
def blocksize_analysis_primary(path):
    data = readfile(path)
    
    obs=data[:,51]
    boot.blocksize_analysis_primary(obs, 500, [2, 500, 10], savefig=1, path=f"{path}/analysis/")
    
def blocksize_analysis_secondary(path):
    data = readfile(path)
    
    def potential(x):
        eps=1e-10
        return -np.log(np.clip(x, eps, None))/5.0 # remember 1/wt
    
    def id(x):
        return x
    
    W = data[:,51]
    
    seed = 8220
    args = [id, W]

    block_vec = [5, 500, 5]

    block_range = range(block_vec[0], block_vec[1], block_vec[2])

    err = []
    
    samples_boot = 200
    for block in block_range:
        pb.progress_bar(block, block_vec[1])
                
        foo, bar = boot.bootstrap_for_secondary(potential, block, samples_boot, 0, args, seed=seed)
        
        err.append(bar)
        # to do: implement error on the standard error
        #tmp = []
        # for sample in range(samples_boot_boot):  
        #     resampling = np.random.randint(0,numblocks,size=numblocks)
        #     W = W[resampling]            
            
        #     args = [id, W]
            
        #     foo, bar = boot.bootstrap_for_secondary(potential, block, samples_boot_boot, 0, args, seed=seed)
        #     tmp.append(bar)

        #d_err.append(0)
        
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
    plt.savefig(f"{path}/analysis/blocksize_analysis_secondary.png", dpi=300, bbox_inches='tight')
          
def plot_potential_Wilsont(path, savefig=0):
    data = readfile(path)
    
    def id(x):
        return x
    
    seed = 8220
    samples = 500
    blocksize = 50
    
    wsmax = 10
    
    wtmaxplot = 7
    wsplot = range(1, wsmax+1)
    
    plt.figure(figsize=(16,12))
    for wt in range(wtmaxplot):
        pb.progress_bar(wt, wtmaxplot)
        V = []
        d_V = []
        
        for ws in range(wsmax):
            W = data[:, 4 + ws + wsmax*wt]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)

            args = [id, W]
            
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, samples, 0, args, seed=seed)
            V.append(ris)
            d_V.append(err)
        
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
    
    def id(x):
        return x
    
    seed = 8220
    
    samples = 500
    blocksize = 50
    
    wtmax = 10
    wsmax = 10
    
    wtmaxplot = 7
    wtplot = np.arange(1, wtmaxplot+1)
    
    plt.figure(figsize=(16,12))
    for ws in range(wsmax):
        V = []
        d_V = []
        
        for wt in range(wtmaxplot):
            pb.progress_bar(wt, wtmax)
            W = data[:, 4 + ws + wtmax*wt]
            
            args = [id, W]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, samples, 0, args, seed=seed)
            V.append(potential(np.mean(W)))
            d_V.append(err)
                    
        plt.errorbar(1/wtplot, V, d_V, **plot.data(ws), label=fr'$w_s={ws+1}$')

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
        plt.savefig(f"{path}/analysis/potential_ws.png", dpi=300, bbox_inches='tight')
        
def plot_fit_potential_ws(path, savefig=0):
    data = readfile(path)
    
    # for the bootstrapping of <W(wt,ws)>
    def id(x): 
        return x
    
    # for fitting aV(wt,ws) for small wt
    def ansatz(x, pars):
        return pars[0] + pars[1]/x
    
    # wrapper for curve_fit which does not support *pars
    def ansatz_wrapper(x, a, b):
        pars = [a, b]
        return ansatz(x, pars)
    
    # seed for the bootstrapping procedure to keep cross correlation intact
    seed = 8220
    
    # samples for bootstrapping
    samples = 500
    # blocksize for blocked bootstrap
    blocksize = 50
    
    # total number of different wt measured
    wtmax = 10
    # total number of different ws measured
    wsmax = 1
    
    # max wt used for extrapolating
    wtmaxplot = 7
        
    x = np.arange(1, wtmaxplot+1)
    
    # stuff that gets printed to file
    V_0_file = []
    d_V_0_file = []
    b_file = []
    d_b_file = []
    chi2red_file = []

    plt.figure(figsize=(16,12))
    for ws in range(wsmax):
        y_t0 = []
        d_y = []
        boot_y = []
            
        for wt in range(wtmaxplot):
            pb.progress_bar(wt, wtmax)
            W = data[:, 4 + ws + wtmax*wt]
            
            args = [id, W]
            
            # aV(wt,ws) = -1/wt*log(<W(wt,ws)>)
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            # computing res and err for aV(wt,ws), keeping the bootstrap samples
            ris, err, bsamples = boot.bootstrap_for_secondary(potential, blocksize, samples, 0, args, seed=seed, returnsamples=1)
            
            y_t0.append(potential(np.mean(W))) # on the original data
            d_y.append(err)
            boot_y.append(bsamples)
        
        y_t0 = np.array(y_t0)
        d_y = np.array(d_y)
        boot_y = np.column_stack(boot_y)
        
        curve_fit_dict = {"p0": [1,1]}
        
        opt, cov, chi2red, boot_opt, boot_cov = reg.fit_yerr_uncorrelated(ansatz_wrapper, x, y_t0, d_y, boot_y, \
        [4,7], [0,7], [1,7], 1, \
        curve_fit_dict, plot.data(ws), plot.fit(ws), plot.conf_band(ws), label=fr'$w_s={ws+1}$')

        V_0_file.append(opt[0])
        d_V_0_file.append(cov[0][0]**0.5)
        b_file.append(opt[0])
        d_b_file.append(cov[1][1]**0.5)
        chi2red_file.append(chi2red)
        
        print(V_0_file)
        print(d_V_0_file)
        print(chi2red_file)

        
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
        plt.savefig(f"{path}/analysis/fit_potential_ws.png", dpi=300, bbox_inches='tight')
        
    x_file = np.arange(1,11)
    data = np.column_stack((x_file, np.array(V_0_file), np.array(d_V_0_file), np.array(b_file), np.array(d_b_file), np.array(chi2red_file)))

    np.savetxt(f"{path}/analysis/output.txt", data)

def plot_fit_potential_ws_CB(path, savefig=0):
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
    blocksize = 50
    # initial guess of parameters
    params = [1,1]         
    
    
    # total number of different wt measured
    wtmax = 10
    # total number of different ws measured
    wsmax = 10
    
    # max wt used for extrapolating
    wtmaxplot = 7
        
    x = np.arange(1, wtmaxplot+1)
    
    # stuff that gets printed to file
    V_0_file = []
    d_V_0_file = []
    b_file = []
    d_b_file = []
    chi2red_file = []

    y_t0 = []
    d_y = []
    
    ws=9
            
    for wt in range(wtmaxplot):
            pb.progress_bar(wt, wtmax)
            W = data[:, 4 + ws + wtmax*wt]
            
            args = [id, W]
            
            # aV(wt,ws) = -1/wt*log(<W(wt,ws)>)
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            # computing res and err for aV(wt,ws), keeping the bootstrap samples
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, samples, 0, args, seed=seed, returnsamples=0)
            
            y_t0.append(potential(np.mean(W))) # on the original data
            d_y.append(err)
        
    y_t0 = np.array(y_t0)
    d_y = np.array(d_y)      
        
        
    ris, err, chi2, dof, pvalue, boot_sample = reg.fit_with_yerr(x, y_t0, d_y, 3, 6, ansatz, params, samples \
                                                                    , save_figs=1, save_path=f'{path}/analysis/extrapolation/ws_{ws+1}'\
                                                                    , plot_title = fr'$aV(w_t, w_s={ws+1})$ for $\beta = 1.7, N_s = 42, N_t = 42$', xlab = r'$w_t$', ylab = r'$aV(w_t)$')
                
    V_0_file.append(ris[0])
    d_V_0_file.append(err[0])
    b_file.append(ris[1])
    d_b_file.append(err[1])
    chi2red_file.append(chi2/dof)        
        
    data = np.column_stack((ws+1, np.array(V_0_file), np.array(d_V_0_file), np.array(b_file), np.array(d_b_file), np.array(chi2red_file)))
    
    np.savetxt(f"{path}/analysis/extrapolation/ws_{ws+1}/fit_results.txt", data)

def concatenate_fit_files(path):
    output_path = f"{path}/analysis/fit_results.txt"

    with open(output_path, 'w') as outfile:
        for i in range(1, 11):
            file_path = f"{path}/analysis/extrapolation/ws_{i}/fit_results.txt"
            with open(file_path, 'r') as infile:
                line = infile.readline()
                outfile.write(line.strip() + '\n') 


def fit_plot_Cornell(path):
    output = np.loadtxt(f'{path}/analysis/fit_results.txt', skiprows=0)
    columns = [output[:, i] for i in range(output.shape[1])]
    
    def Cornell(x, pars):
        return pars[0] + pars[1]*x - pars[2]/x
    
    x=columns[0]
    y=columns[1]
    dy=columns[2]
    
    numsamples = 1000
    
    fitparams=np.array([0.1, 1., 1.2], dtype=np.float64)
            
    ris, err, chi2, dof, pvalue, boot_sample = reg.fit_with_yerr(x, y, dy, 3, 8, Cornell, fitparams, numsamples,\
                                                                    save_figs=1, save_path=f'{path}/regression/analysis/Cornell'\
                                                                    , plot_title = fr'$aV(w_s)$ for $\beta = 1.7, N_s = 42, N_t = 42$', xlab = r'$w_t$', ylab = r'$aV(w_t)$')
    print("  chi^2/dof = {:3.3f}/{:d} = {:3.3f}".format(chi2, dof, chi2/dof))
    print("  p-value = %f" % pvalue)
    print()
    print("  alfa = {: f} +- {:f}".format(ris[0], err[0]) ) 
    print("  sigma = {: f} +- {:f}".format(ris[1], err[1]) )
    print("  c = {: f} +- {:f}".format(ris[2], err[2]) ) 
    print()    

if __name__ == "__main__":
    path = "/home/negro/projects/matching/string_tension/L42_b1.7"
    
    #blocksize_analysis_secondary(path)
    
    #plot_potential_Wilsont(path, savefig=1)
    
    #plot_fit_potential_ws_CB(path, savefig=1)
    
    #concatenate_fit_files(path)
    
    #fit_plot_Cornell(path)