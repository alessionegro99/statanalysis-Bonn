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

def readfile(path):
    output = np.loadtxt(f'{path}/analysis/dati.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    return np.column_stack(columns)
    
def thermalization(path, wt_max, ws_max):
    os.makedirs(f'{path}/analysis/therm/', exist_ok=True)
                  
    for ws in range(0,ws_max, ws_max//2):
        for wt in range(0, wt_max, wt_max//2):
            output = np.loadtxt(f'{path}/analysis/dati.dat', skiprows=1)
            Wloop_wt_ws = output[:, 4 + ws + ws_max*wt]
    
            x = np.arange(0,len(Wloop_wt_ws), len(Wloop_wt_ws)//500)
            y = Wloop_wt_ws[0:len(Wloop_wt_ws):len(Wloop_wt_ws)//500]

            plt.figure(figsize=(18,12))
            plt.plot(x, y
                    , marker = 'o'
                    , linestyle = '-', linewidth = 0.375
                    , markersize = 2
                    , color = plot.color_dict[1])
            
            plt.xlabel(r'$t_i$')

            plt.grid (True, linestyle = '--', linewidth = 0.25)

            plt.savefig(f"{path}/analysis/therm/therm_wt{wt}_ws{ws}.png", bbox_inches='tight')
            plt.close()
        
def bsa_potential(path, wt_max, ws_max):
    os.makedirs(f'{path}/analysis/bsa/', exist_ok=True)
        
    data = readfile(path)
    
    for ws in range(0,ws_max, ws_max//2):
        for wt in range(0, wt_max, wt_max//2):
            Wloop = data[:, 4 + ws + ws_max*wt]

            boot.blocksize_analysis_primary(Wloop, 200, [10, len(Wloop)//10, 10], savefig=1, path=f"{path}/analysis/bsa", extra=f'wt{wt}_ws{ws}')
            
def boot_fit(x, y, d_y, b_y, model, lim_inf, lim_sup):
    x_fit, y_fit, d_y_fit, b_y_fit = x[lim_inf:lim_sup], y[lim_inf:lim_sup], d_y[lim_inf:lim_sup], b_y[lim_inf:lim_sup]
    opt, cov = curve_fit(model, x_fit, y_fit, sigma=d_y_fit, absolute_sigma=True)
    
    n_boot = len(b_y[0])

    x_linsp = np.linspace(x_fit[0], x_fit[-1], 100)
    y_linsp = model(x_linsp, *opt)
    
    b_opt = []
    b_c2r = []
    b_y_linsp = []
    
    for j in range(n_boot):
        y_fit_j = [b_y_fit[i][j] for i in range(len(b_y_fit))] 
        
        opt_j, cov_j = curve_fit(model, x_fit, y_fit_j, sigma=d_y_fit, absolute_sigma=True)
        
        b_c2r.append(reg.chi2_corr(x_fit, y_fit_j, model, np.diag(np.array(d_y_fit)**2), *opt_j))
        
        b_opt.append(opt_j)
        
        y_linsp_j = model(x_linsp, *opt_j)
        
        b_y_linsp.append(y_linsp_j)
    
        d_opt = np.std(b_opt, axis=0, ddof=1)
    
    c2r = reg.chi2_corr(x_fit, y_fit, model, np.diag(np.array(d_y_fit)**2), *opt)
    d_c2r = np.std(b_c2r, ddof=1)
    
    d_y_linsp = np.std(b_y_linsp, axis=0, ddof=1)
    
    return x_linsp, y_linsp, d_y_linsp, opt, d_opt, b_opt, c2r, d_c2r              

def get_potential (path, wt_max, ws_max):

    data = readfile(path)
    def id(x):
        return x

    wsplot = np.arange(1,ws_max+1)

    seed = 8220
    samples = 500
    blocksizes = [500] * len(wsplot) 
                
    V = []
    d_V = []
    V_bs = []
    
    for ws, blocksize in zip(range(ws_max), blocksizes):
        V_ws = []
        d_V_ws = []
        V_bs_ws = []
        
        for wt in range(wt_max):
            W = data[:, 4 + ws + ws_max*wt]

            args = [id, W]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            _, err, bs = boot.bootstrap_for_secondary(potential, blocksize, samples, 0, args, seed=seed, returnsamples=1)
        
            sys.stdout.write(f"Currently bootstrapping w_t={wt+1}/{wt_max}, w_s={wsplot[ws]:.2f}.\n")
            sys.stdout.flush()

            V_ws.append(potential(np.mean(W)))
            d_V_ws.append(err)
            V_bs_ws.append(bs)
        
        V.append(V_ws)
        d_V.append(d_V_ws)
        V_bs.append(V_bs_ws)
        
    np.save(f"{path}/analysis/potential_wt", np.array([wsplot, V, d_V, V_bs], dtype=object))
            
def plot_potential(path, wt_max_plot, ws_max_plot):
    data = np.load(f"{path}/analysis/potential_wt.npy", allow_pickle=True)
    
    wsplot, res, d_res, res_bs = map(np.array, data[:4])
    
    res = np.transpose(res)
    d_res = np.transpose(d_res)
    res_bs = np.transpose(res_bs)
            
    ## effective mass plot 
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$')

    for i in range(ws_max_plot):
        effm = res[i]
        d_effm = d_res[i]
        plt.errorbar(np.arange(1,wt_max_plot[i]+1), effm[:wt_max_plot[i]], d_effm[:wt_max_plot[i]], **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
                
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/effmass.png", dpi=300, bbox_inches='tight')
    
    ## effective mass plot vs 1/wt 
    plt.figure(figsize = (18,12))
    
    plt.xlabel(r'$1/w_t$')
    plt.ylabel(r'$aV(w_t)$')

    for i in range(ws_max_plot):
        effm = res[i]
        d_effm = d_res[i]
        plt.errorbar(1/np.arange(1,wt_max_plot[i]+1), effm[:wt_max_plot[i]], d_effm[:wt_max_plot[i]], **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
                
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/pot_ws.png", dpi=300, bbox_inches='tight')

def find_wtmin(path, ws_max_plot, wt_start=1, wt_end=10):
    def model(x, a, b):
        return a + b/x
    
    data = np.load(f"{path}/analysis/potential_wt.npy", allow_pickle=True)
    
    res, d_res, boot_res = map(np.array, data[1:4])
    
    res = np.transpose(res)
    d_res = np.transpose(d_res)
    boot_res = np.transpose(boot_res)
    
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_{t,min}$')
    plt.ylabel(r'$aV(w_t=\infty)$')
    for i in range(ws_max_plot):
        dp = []
        d_dp = []
        for wt_min_fit in range(wt_start,wt_end-2):
            mf, Mf = wt_min_fit, wt_end
            x_fit, y_fit, d_y_fit = np.arange(mf,Mf), np.array(res[i][mf:Mf]), np.array(d_res[i][mf:Mf])
                    
            opt, _ = curve_fit(model, x_fit, y_fit, sigma=d_y_fit, absolute_sigma=True)
            
            boot_y_fit = np.array(boot_res[i][mf:Mf])
            n_boot = np.shape(boot_y_fit)[1]
            
            boot_pot = []

            for j in range(n_boot):
                y_fit = boot_y_fit[:,j]
                
                opt_j, _ = curve_fit(model, x_fit, y_fit, sigma=d_y_fit, absolute_sigma=True)
                
                boot_pot.append(opt_j[0])
            
            pot, d_pot = opt[0], np.std(boot_pot, ddof=1)
            dp.append(pot)
            d_dp.append(d_pot)
        
        plt.errorbar(range(wt_start, wt_end-2), dp, d_dp, **plot.data(i), label=i)
           
    plt.legend
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"{path}/analysis/find_wtmin.png", dpi=300, bbox_inches='tight')    

def extrap_potential(path, wt_max_plot, ws_max_plot, wt_min_fit, wt_max_fit):
    def model(x, a, b):
        return a + b*x
    
    data = np.load(f"{path}/analysis/potential_wt.npy", allow_pickle=True)
    
    wsplot, res, d_res, boot_res = map(np.array, data[:4])
        
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$')
    
    pot, b_pot = [], []
            
    print("ws pot_n d_pot_n d_pot_b c2r_n c2r_b d_c2r_b")
    for i in range(ws_max_plot):
        
        Mp = wt_max_plot[i]
        mf, Mf = wt_min_fit[i], wt_max_fit[i]
        
        x, y, d_y, b_y = 1/np.arange(1,Mp+1), np.array(res[i][0:Mp]), np.array(d_res[i][0:Mp]), np.array(boot_res[i][0:Mp])
        
        x_linsp, y_linsp, d_y_linsp, opt, d_opt, b_opt, c2r, d_c2r = boot_fit(x, y, d_y, b_y, model, mf, Mf)
        
        pot.append(opt[0])
        b_pot.append([b_opt[j][0] for j in range(len(b_opt))])
        
        plt.errorbar(x, y, d_y, **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$, $\chi^2/dof$={c2r:.2f}")

        plt.plot(x_linsp, y_linsp, **plot.fit(i))
        plt.fill_between(x_linsp, y_linsp-d_y_linsp, y_linsp+d_y_linsp, **plot.conf_band(i))

        print(f"{wsplot[i]:.2g} {opt[0]:.6g} {d_opt[0]:.2g} {c2r:.4g} {d_c2r:.4g}")

    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    plt.savefig(f"{path}/analysis/extrap_pot.png", dpi=300, bbox_inches='tight')    

    wsplot = wsplot[:ws_max_plot].copy()
    save = np.array([wsplot, pot, b_pot], dtype=object)
    np.save(f"{path}/analysis/opt", save)       
  
## main

def planarpot(path):    
    wt_max = 10
    ws_max = 10
    
    if not os.path.isdir(f"{path}/analysis/therm/"):
        thermalization(path, wt_max, ws_max)
        
    if not os.path.isdir(f"{path}/analysis/bsa/"):
        bsa_potential(path, wt_max, ws_max)
    
    ws_max_plot = ws_max
    wt_max_plot = [wt_max]*ws_max_plot
    
    if not os.path.isfile(f"{path}/analysis/potential_wt.npy"):
           get_potential(path, wt_max, ws_max)
    
    plot_potential(path, wt_max_plot, ws_max_plot)
    
    if not os.path.isfile(f"{path}/analysis/find_wtmin.png"):
        find_wtmin(path, ws_max_plot, wt_start=1, wt_end=10)

    wt_min_fit = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    wt_max_fit = [10] * ws_max_plot
    
    extrap_potential(path, wt_max_plot, ws_max_plot, wt_min_fit, wt_max_fit)
    
def plotfit_potential(path, fit='false'):   
    x, y, b_y = np.load(f"{path}/analysis/opt.npy", allow_pickle=True)
    
    d_y = [np.std(b_y[i], ddof=1) for i in range(len(b_y))]  

    plt.figure(figsize=(18,12))
    plt.errorbar(x, y, d_y, **plot.data(0))
    plt.xlabel(r"r/a")
    plt.ylabel(r"aV(r/a)")
    
    if(fit == 'true'):
        start = 3
        end = 10
                        
        def potential(x, a, b, c, d):
            return a + b*x + c*np.log(x) + d/x
 
        x_linsp, y_linsp, d_y_linsp, opt, d_opt, b_opt, c2r, d_c2r = boot_fit(x, y, d_y, b_y, potential, start, end)
           
        plt.plot(x_linsp, y_linsp, **plot.fit(4), label=fr"$\sigma$={opt[1]:.4g}({d_opt[1]:.4g})")
        plt.fill_between(x_linsp, y_linsp-d_y_linsp, y_linsp+d_y_linsp, **plot.conf_band(4))

        print('sigma d_sigma d_sigma_boot c2r')
        print(f'{opt[1]:.6g} {d_opt[1]:.4g} {c2r:.4f} {d_c2r:.4f}')    
        
        plt.legend()

    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"{path}/analysis/potential.png", dpi=300, bbox_inches='tight')    

if __name__ == "__main__":
    path_glob = f"/home/negro/projects/matching/step_scaling/infvol_PBC/T42_L42_b5"
    
    concatenate.concatenate(f"{path_glob}/data", 2000, f"{path_glob}/analysis/")
    
    ## planar
    planarpot(path_glob)

    plotfit_potential(path_glob, fit='true')