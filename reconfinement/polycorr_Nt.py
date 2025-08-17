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

    # plaqs plaqt polyre polyim xi0 ximin TrP2 ReG0 ImG0 ReG1 ImG1 ... ReG23 ImG23
    
    return np.column_stack(columns)

def thermalization(path): 
    data = readfile(path)
    
    idx = 1
    
    plaqt = data[:,4 + idx*2]
    
    x = np.arange(0,len(plaqt), len(plaqt)//500)
    y = plaqt[0:len(plaqt):len(plaqt)//500]

    plt.figure(figsize=(16,12))
    plt.plot(x, y
             , marker = 'o'
             , linestyle = '-', linewidth = 0.375
             , markersize = 2
             , color = plot.color_dict[1])
    
    plt.xlabel(r'$t_i$')
    #plt.ylabel(r'$G(1)(t_i)$', rotation = 0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$MC history$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.savefig(f"{path}/analysis/thermalization_G({idx}).png", dpi=300, bbox_inches='tight')
    
def blocksize_analysis_primary(path):
    print("reading file...")
    data = readfile(path)
    print("read file ")
    
    for aux in [7]:
        obs=data[:,aux]
        boot.blocksize_analysis_primary(obs, 200, [2, 10000, 50], savefig=1, path=f"{path}/analysis/")
 
def polycorr(path):
    data = readfile(path)
    
    def id(x):
        return x
    
    maxpolycorr = 48
    seed = 8220
    samples = 1000
    blocksize = 2000
    
    x = np.arange(0, maxpolycorr)   
    y = []
    d_y = []

    for i in range(maxpolycorr):
        pb.progress_bar(i, maxpolycorr)
        tmp = data[:, 7 + 2*i]
        
        _, err = boot.bootstrap_for_primary(id, tmp, blocksize, samples, seed)
        
        y.append(np.mean(tmp))
        d_y.append(err)
    
    plt.figure(figsize = (16,12))
    
    plt.errorbar(x, y, d_y, **plot.data(1))
    
    plt.xlabel(r'$R$')
    plt.ylabel(r'$G(R)$', rotation = 0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
            
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    
    plt.yscale('log')
    
    plt.savefig(f"{path}/analysis/polycorr.png", dpi=300, bbox_inches='tight')
    
    print(d_y[0])
        
def fit_polycorr(path):
    data = readfile(path)
    
    def id(x):
        return x
    
    def sym_k0(x, a, E0):
        return a*(k0(E0*x) + k0(E0*(96-x)))
    
    maxpolycorr = 48
    seed = 8220
    samples = 1000
    blocksize = 2000
    xmin = 20
    xmax = 48
    
    y_t0 = []
    
    y = []
    d_y = []
        
    x = np.arange(0, maxpolycorr)      
    mask = ((x<=xmax) & (x>=xmin))

    for i in range(maxpolycorr):
        pb.progress_bar(i, maxpolycorr)
        tmp = data[:, 7 + 2*i]
        
        _, err = boot.bootstrap_for_primary(id, tmp, blocksize, samples, seed)
        
        y_t0.append(tmp)
        y.append(np.mean(tmp))
        d_y.append(err)
    
    y_t0 = np.array(y_t0).T
    y = np.array(y)
    d_y = np.array(d_y)
    
    ## fitting
    x_mskd = x[mask]
    y_t0_mskd = y_t0[:,mask]
    y_mskd = y[mask]
    d_y_mskd = d_y[mask]
    
    R_ij = np.corrcoef(y_t0_mskd, rowvar=False)
    
    p0 = np.asarray([1, 1], float)
    bounds = 0, np.inf
    
    C_ij = np.outer(d_y_mskd, d_y_mskd) * R_ij # first compute R_ij then correct for autocorrelation (sigma_ii naively estimated on the MC history is not correct due to autocorrelation!)
        
    opt, cov = curve_fit(sym_k0, x_mskd, y_mskd, sigma=C_ij, absolute_sigma=True, p0=p0, bounds=bounds)
    
    chi2 = reg.chi2_corr(x_mskd, y_mskd, sym_k0, C_ij, opt[0], opt[1])
    chi2red = chi2/(len(x_mskd) - 2)
    
    print(opt)
    print(cov**0.5)
    print(chi2red) 
    
    ## plotting
    x_fit = np.linspace(x_mskd[0],x_mskd[-1],500)
    y_fit = sym_k0(x_fit, opt[0], opt[1])
    
    plt.figure(figsize = (16,12))
    
    plt.errorbar(x_mskd, y_mskd, d_y_mskd, **plot.data(1))
    
    plt.plot(x_fit, y_fit, **plot.fit(1))
        
    plt.xlabel(r'$R$')
    plt.ylabel(r'$G(R)$', rotation = 0)
    plt.gca().yaxis.set_label_coords(-0.15, 0.5)
    
    boot_opt = np.random.multivariate_normal(opt, cov, size=samples, tol=1e-10)
        
    boot_y_fit = [sym_k0(x_fit, boot_opt[i,0], boot_opt[i,1]) for i in range(samples)]        
    boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
    plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(1))

    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    
    plt.yscale('log')
    
    plt.savefig(f"{path}/analysis/fit_polycorr.png", dpi=300, bbox_inches='tight')
    
    data = [xmin, xmax, opt[0], cov[0,0]**0.5, opt[1], cov[1,1]**0.5, chi2red]
    np.savetxt(f"{path}/analysis/fit_res.txt", data)
    
def boot_fit_polycorr(path):
    data = readfile(path)
    
    def id(x):
        return x
    
    def sym_k0(x, a, E0):
        return a*(k0(E0*x) + k0(E0*(96-x)))
    
    maxpolycorr = 24
    seed = 8220
    samples = 1000
    blocksize = 2000
    xmin = 15
    xmax = 23
    
    x = np.arange(0, maxpolycorr)      
    mask = ((x<=xmax) & (x>=xmin))
    
    num_blocks = len(data[:,0])//blocksize
    
    opt_list = []
    chi2red_list = []
    eigv_list = []
            
    p0 = np.asarray([1, 1], float)
    bounds = 0, np.inf
    
    for sample in range(samples):
        pb.progress_bar(sample, samples)

        y_t0 = []
        y = []
        d_y = []
        
        rng = np.random.default_rng(seed + sample)
        idx_blocks = rng.integers(0, num_blocks, size=num_blocks)
        
        for i in range(maxpolycorr):
            tmp = data[:, 7 + 2*i]
            
            # drop last elements
            tmp = tmp[:(num_blocks*blocksize)]
                        
            # shuffle blockwise with repetition (remember C_ij needs unblocked data)
            tmp = np.concatenate([tmp[(j*blocksize):((j+1)*blocksize)] for j in idx_blocks])
            
            y_t0.append(tmp) # unblocked data for C_ij
            y.append(np.mean(tmp)) 
            
            tmp_blockmean = []
            for j in range(num_blocks):
                tmp_blockmean.append(np.mean(tmp[(j*blocksize):((j+1)*blocksize)]))
            
            d_y.append(np.std(tmp_blockmean)/num_blocks**0.5) # \tau_int correction
        
        y_t0 = np.array(y_t0).T
        y = np.array(y)
        d_y = np.array(d_y)
                
        ## fitting
        x_mskd = x[mask]
        y_t0_mskd = y_t0[:,mask]
        y_mskd = y[mask]
        d_y_mskd = d_y[mask]
        
        R_ij = np.corrcoef(y_t0_mskd, rowvar=False)
        
        # first compute R_ij then correct for autocorrelation (sigma_ii naively estimated on the MC history is not correct due to autocorrelation!)
        C_ij = np.outer(d_y_mskd, d_y_mskd) * R_ij
        
        eigv_list.append(np.min(eig(C_ij)[0]).real)
 
        opt, cov = curve_fit(sym_k0, x_mskd, y_mskd, sigma=C_ij, absolute_sigma=True, p0=p0, bounds=bounds)
        chi2 = reg.chi2_corr(x_mskd, y_mskd, sym_k0, C_ij, opt[0], opt[1])
        chi2red = chi2/(len(x_mskd) - 2)
        
        chi2red_list.append(chi2red)
        
        p0 = opt
        
        opt_list.append(opt)
        
    eigv_list = np.array(eigv_list)        
    opt_list = np.array(opt_list)
    
    boot_opt = np.mean(opt_list[:,1])
    boot_opt_std = np.std(opt_list[:,1], ddof=1)
    
    boot_chi2red = np.mean(chi2red_list)
    boot_chi2red_std = np.std(chi2red_list, ddof=1)
    
    boot_eigv = np.mean(eigv_list)
    boot_eigv_std = np.std(eigv_list, ddof=1)

    print(boot_opt, boot_opt_std)
    print(boot_chi2red, boot_chi2red_std)
    print(boot_eigv, boot_eigv_std)
    
    data = [boot_opt, boot_opt_std, boot_chi2red, boot_chi2red_std, boot_eigv, boot_eigv_std]
    np.savetxt(f"{path}/analysis/boot_fit_res.txt", data)

def format_results():
    beta = 23.3805
    h = 0.005
    
    path = f"/home/negro/projects/reconfinement/polycorr_Nt/b{beta}_h{h}"
    # Rmin Rmax A d_A boot_A d_boot_A E0 d_E0 boot_E0 d_boot_E0 chi2 boot_chi2 d_boot_chi2 lambda0 d_lambda0
    
    header_line = ["Nt", "Rmin", "Rmax", "A", "d_A", "E0", "d_E0", "boot_E0", "d_boot_E0", "chi2red", "boot_chi2red", "d_boot_chi2red"]
    header_line = " ".join(header_line)
    
    with open(f"{path}/results_b{beta}_h{h}.txt", "w") as f:
        f.write(header_line + "\n")
    
    for Nt in [16]:
        fit_res = []
        with open(f"{path}/corr_48_{Nt}_96_{h}/analysis/fit_res.txt", "r") as f:
            for line in f:
                fit_res.append(float(line))
                    
        boot_fit_res = []
        with open(f"{path}/corr_48_{Nt}_96_{h}/analysis/boot_fit_res.txt", "r") as f:
            for line in f:
                boot_fit_res.append(float(line))
                
        formatted_result = [Nt, fit_res[0], fit_res[1], fit_res[2], fit_res[3], fit_res[4], fit_res[5], boot_fit_res[0], boot_fit_res[1], fit_res[6], boot_fit_res[2], boot_fit_res[3]]
        
        with open(f"{path}/results_b{beta}_h{h}.txt", "a") as f:
            f.write(" ".join(map(str, formatted_result)) + "\n")
    
if __name__ == "__main__":
    
    for Ns in [9]:
        path = f"/home/negro/projects/reconfinement/polycorr_Nt/b23.3805_h0.007/corr_24_{Ns}_96_0.007"
                
                
        #concatenate.concatenate(f"{path}/data", 2000, f"{path}/analysis")
        
        #thermalization(path)
        #blocksize_analysis_primary(path)
        
        #polycorr(path)

        fit_polycorr(path)
        
        #boot_fit_polycorr(path)
        
    #format_results()