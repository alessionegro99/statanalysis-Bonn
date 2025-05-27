import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.special import k0

import concatenate
import plot

import progressbar as pb
import bootstrap as boot
import regression as reg

def readfile(path, skiprows = 1):
    output = np.loadtxt(f'{path}/analysis/dati.dat', skiprows=skiprows)
    columns = [output[:, i] for i in range(output.shape[1])]

    # plaqs plaqt polyre polyim ReG0 ImG0 ReG1 ImG1 ... ReG23 ImG23
    
    return np.column_stack(columns)

def thermalization(path): 
    data = readfile(path)
    
    plaqt = data[:,6]
    
    x = np.arange(0,len(plaqt), len(plaqt)//500)
    y = plaqt[0:len(plaqt):len(plaqt)//500]

    plt.figure(figsize=(16,12))
    plt.plot(x, y
             , marker = 'o'
             , linestyle = '-', linewidth = 0.375
             , markersize = 2
             , color = plot.color_dict[1])
    
    plt.xlabel(r'$t_i$')
    plt.ylabel(r'$G(1)(t_i)$', rotation = 0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'MC history of $G(R=1)$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.savefig(f"{path}/analysis/thermalization.png", dpi=300, bbox_inches='tight')
    
def blocksize_analysis_primary(path):
    print("reading file...")
    data = readfile(path)
    print("read file ")
    
    for aux in [44]:
        obs=data[:,aux]
        boot.blocksize_analysis_primary(obs, 200, [2, 10000, 50], savefig=1, path=f"{path}/analysis/")
 
def polycorr(path):
    data = readfile(path)
    
    def id(x):
        return x
    
    maxpolycorr = 24
    seed = 8220
    samples = 1000
    blocksize = 2000
    
    x = np.arange(0, maxpolycorr)   
    y = []
    d_y = []

    for i in range(maxpolycorr):
        pb.progress_bar(i, maxpolycorr)
        tmp = data[:, 4 + 2*i]
        
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
        
def fit_polycorr(path):
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
    
    y_t0 = []
    
    y = []
    d_y = []
        
    x = np.arange(0, maxpolycorr)      
    mask = ((x<=xmax) & (x>=xmin))

    for i in range(maxpolycorr):
        pb.progress_bar(i, maxpolycorr)
        tmp = data[:, 4 + 2*i]
        
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
    
    data = [xmin, xmax, opt[0], opt[1], cov[0,0], cov[1,0], cov[0,1], cov[1,1], chi2red]
    np.savetxt(f"{path}/analysis/fit_res.txt", data)
    
def boot_fit_polycorr(path):

    def model(x, a, E0):
        return _fit.sym_K0(x, 96, a, E0)
    
    ## fit on the original sample
    ReG_t0 = df[ReG].to_numpy()
    
    x = np.arange(15,24)
    
    y_t0 = ReG_t0[:, x]
    y = np.mean(y_t0, axis = 0)
        
    p0 = np.asarray([1, 1], float)
    bounds = 0, np.inf
    
    R_ij = np.corrcoef(y_t0, rowvar=False)
        
    b_y_t0 = np.apply_along_axis(_bootstrap.blocking, axis=0, arr=y_t0, block_size=4000, discard_end=True)
    
    b_t0 = b_y_t0.shape[0]
    
    err = np.std(b_y_t0, axis = 0)/b_t0**0.5
    
    C_ij = np.outer(err, err) * R_ij # first compute R_ij then correct for autocorrelation (sigma_ii naively estimated on the MC history is not correct due to autocorrelation!)
    
    opt, cov = curve_fit(model, x, y, sigma=C_ij, absolute_sigma=True, p0=p0, bounds=bounds)
    
    chi2 = _fit.chi2_yerr(x, y, model, C_ij, opt[0], opt[1])
    
    print(opt)
    print(cov**0.5)
    print(chi2)
    
    n_col = np.shape(y_t0)[1]
    
    n_boot = 500
    
    chi_list = []
            
    for n in range(n_boot):
        _progressbar.progress_bar(n, n_boot)
        y_t = []
        b_y_t = []
        
        seed = 423784

        for i in range(n_col):
            y_t.append(_bootstrap.bootstrap_samples(y_t0[:,i], 1, seed + n))
        
        y_t = np.array(y_t)
        y_t = np.squeeze(y_t).T
                                                 
        y = np.mean(y_t, axis = 0)
                
        R_ij = np.corrcoef(y_t, rowvar=False)
          
        for i in range(n_col):
            b_y_t.append(_bootstrap.bootstrap_samples(b_y_t0[:,i], 1, seed + n))
        
        b_y_t = np.array(b_y_t)
        b_y_t = np.squeeze(b_y_t).T
        
        b_t = b_y_t.shape[0]
                
        err = np.std(b_y_t, axis = 0)/b_t**0.5
        
        C_ij = np.outer(err, err) * R_ij # first compute R_ij then correct for autocorrelation (sigma_ii naively estimated on the MC history is not correct due to autocorrelation!)
                
        opt, cov = curve_fit(model, x, y, sigma=C_ij, absolute_sigma=True, p0=p0, bounds=bounds, maxfev = 1000)
        
        chi2 = _fit.chi2_yerr(x, y, model, C_ij, opt[0], opt[1])
        
        p0=opt
        
        chi_list.append(chi2)
            
    
    print(np.mean(chi_list), np.std(chi_list))

if __name__ == "__main__":
    
    path = "/home/negro/projects/reconfinement/polycorr_Nt/b23.3805_h0.005/corr_24_9_96_0.005"
    
    #concatenate.concatenate(f"{path}/rawdata", 2000)
    
    #thermalization(path)
    #blocksize_analysis_primary(path)
    
    #polycorr(path)

    fit_polycorr(path)