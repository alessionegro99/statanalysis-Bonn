from lib import _plot
from lib import _bootstrap
from lib import _progressbar
from lib import _fit

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import k0


def thermalization(path):
    
    new_columns = ["Us", "Ut", "ReP", "ImP"]

    # add ReG0, ImG0, ReG1, ImG1, ..., ReG23, ImG23
    for i in range(24):
        new_columns.append(f"ReG{i}")
        new_columns.append(f"ImG{i}")
    new_columns.append("null")
    
    df = pl.read_csv(f"{path}dati.dat"
                     , has_header=False
                     , skip_rows = 1
                     , separator=" "
                     , new_columns=new_columns)
    df = df.drop(["null"])
    
    ReG = []
    for i in range(24):
        ReG.append(f"ReG{i}")    
    
    ReG_t0 = df[ReG].to_numpy()
    
    ReG1_t0 = ReG_t0[:,1]
     
    x = np.arange(0,len(ReG1_t0), len(ReG1_t0)//500)
    y = ReG1_t0[x]

    plt.figure(figsize=(16,9))
    plt.plot(x, y
             , marker = 'o'
             , linestyle = '-', linewidth = 0.375
             , markersize = 2
             , color = _plot.color_dict[1])
    
    plt.xlabel(r'$t_i$')
    plt.ylabel(r'$G_1(t_i)$', rotation = 0)
    plt.title(r'MC history of $ReG(R=1)$ for $\beta=23.3805$, $h=0.005$, $N_t=9$, $N_s=96$.')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, which = 'both', linestyle = '--', linewidth = 0.25)

    plt.savefig("thermalization.png", dpi=300, bbox_inches='tight')
    
def binning_analysis(path):
    new_columns = ["Us", "Ut", "ReP", "ImP"]

    # add ReG0, ImG0, ReG1, ImG1, ..., ReG23, ImG23
    for i in range(24):
        new_columns.append(f"ReG{i}")
        new_columns.append(f"ImG{i}")
    new_columns.append("null")
    
    df = pl.read_csv(f"{path}dati.dat"
                     , has_header=False
                     , skip_rows = 1
                     , separator=" "
                     , new_columns=new_columns)
    df = df.drop(["null"])
    
    ReG = []
    for i in range(24):
        ReG.append(f"ReG{i}")    
    
    ReG_t0 = df[ReG].to_numpy()
    
    ReG1_t0 = ReG_t0[:,1]
     
    x = np.arange(0,len(ReG1_t0), len(ReG1_t0)//500)
    y = ReG1_t0[x]
    
    fig = _bootstrap.bootstrap_analysis(t0 = ReG1_t0
                                  , n_samples = 200
                                  , block_size_0 = 5
                                  , block_size_step = 100
                                  , seed = 0)
    
    fig.savefig("binning_analysis.png", dpi = 300, bbox_inches = 'tight')
    
def plot_polycorr_nofit(path):
    new_columns = ["Us", "Ut", "ReP", "ImP"]

    # add ReG0, ImG0, ReG1, ImG1, ..., ReG23, ImG23
    for i in range(24):
        new_columns.append(f"ReG{i}")
        new_columns.append(f"ImG{i}")
    new_columns.append("null")
    
    df = pl.read_csv(f"{path}dati.dat"
                     , has_header=False
                     , skip_rows = 1
                     , separator=" "
                     , new_columns=new_columns)
    df = df.drop(["null"])
    
    ReG = []
    for i in range(24):
        ReG.append(f"ReG{i}")    
    
    ReG_t0 = df[ReG].to_numpy()
    
    x = np.arange(0, np.shape(ReG_t0)[1])
    y = np.mean(ReG_t0, 0)
    
    d_y = []
    
    for i in range(np.shape(ReG_t0)[1]):
        b_ReG_i_t0 = _bootstrap.blocking(ReG_t0[:,i], 2000, False)
        d_y.append(np.std(b_ReG_i_t0)/len(b_ReG_i_t0)**0.5)        
    
    plt.figure(figsize=(16,9))
    plt.errorbar(x, y, d_y, **_plot.data(1), label=r'$G(R)=\frac{1}{2Ns^2}\sum_{\vec{x}}\langle P(\vec{x})P(\vec{x}+R)\rangle$')
    plt.yscale('log')
    plt.xlabel(r'$R$')
    plt.ylabel(r'$G(R)$', rotation=0)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.savefig("polycorr_nofit.png", dpi=600, bbox_inches='tight')

def plot_fit_polycorr(path):
    Rmax = 24
    
    #########################
    new_columns = ["Us", "Ut", "ReP", "ImP"]
    
    # add ReG0, ImG0, ReG1, ImG1, ..., ReG23, ImG23
    for i in range(Rmax):
        new_columns.append(f"ReG{i}")
        new_columns.append(f"ImG{i}")
    new_columns.append("null")
    
    df = pl.read_csv(f"{path}dati.dat"
                     , has_header=False
                     , skip_rows = 1
                     , separator=" "
                     , new_columns=new_columns)
    df = df.drop(["null"])
    
    ReG = []
    for i in range(24):
        ReG.append(f"ReG{i}")    
    
    #########################
    
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
    
    # plotting
    
    x_fit = np.linspace(1, Rmax-1, 100)
    y_fit = model(x_fit, opt[0], opt[1])
    y_fit_err = np.sqrt(np.diag(cov))

    x_plot = np.arange(24)
    y_plot = np.mean(ReG_t0, axis = 0)
    d_y_plot = []
    for i in range(np.shape(ReG_t0)[1]):
        b_ReG_i_t0 = _bootstrap.blocking(ReG_t0[:,i], 4000, False)
        d_y_plot.append(np.std(b_ReG_i_t0)/len(b_ReG_i_t0)**0.5)    
    
    plt.figure(figsize=(16,9))
    plt.errorbar(x_plot, y_plot, yerr=d_y_plot, **_plot.data(1), label=r'$G(R)=\frac{1}{2Ns^2}\sum_{\vec{x}}\langle P(\vec{x})P(\vec{x}+R)\rangle$')
    
    plt.plot(x_fit, y_fit, **_plot.fit(1), label=r'$f(R; a, E_0)=K_0(E_0R)-K_0(E_0(N_s-R))$')
    plt.fill_between(x_fit,
                 model(x_fit, opt[0] - y_fit_err[0], opt[1] - y_fit_err[1]),
                 model(x_fit, opt[0] + y_fit_err[0], opt[1] + y_fit_err[1]),
                 **_plot.conf_band(1))
    
    plt.yscale('log')
    plt.xlabel(r'$R$')
    plt.ylabel(r'$G(R)$', rotation=0)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.savefig("polycorr_fit.png", dpi=600, bbox_inches='tight')
    
def fit_polycorr_boot(path):
    Rmax = 24
    
    #########################
    new_columns = ["Us", "Ut", "ReP", "ImP"]
    
    # add ReG0, ImG0, ReG1, ImG1, ..., ReG23, ImG23
    for i in range(Rmax):
        new_columns.append(f"ReG{i}")
        new_columns.append(f"ImG{i}")
    new_columns.append("null")
    
    df = pl.read_csv(f"{path}dati.dat"
                     , has_header=False
                     , skip_rows = 1
                     , separator=" "
                     , new_columns=new_columns)
    df = df.drop(["null"])
    
    ReG = []
    for i in range(24):
        ReG.append(f"ReG{i}")    
    
    #########################
    
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

    
    