import sys
import os
import math

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

## generic
def format(y, d_y, digits=2):
    d_digits = int(math.floor(math.log10(abs(d_y))))
    d_y_rounded = round(d_y, -d_digits + (digits - 1))

    # Determine decimal places needed
    decimal_places = -d_digits + (digits - 1)

    # Round the value to match
    value_rounded = round(y, decimal_places)

    # Format the string
    formatted = f"{value_rounded:.{decimal_places}f}({int(d_y_rounded * 10**decimal_places):0{digits}d})"
    print(formatted)

def boot_fit(x, y, d_y, b_y, model, lim_inf, lim_sup, extension=None):
    x_fit, y_fit, d_y_fit, b_y_fit = x[lim_inf:lim_sup], y[lim_inf:lim_sup], d_y[lim_inf:lim_sup], b_y[lim_inf:lim_sup]
    opt, cov = curve_fit(model, x_fit, y_fit, sigma=d_y_fit, absolute_sigma=True)
    
    n_boot = len(b_y[0])

    x_linsp = np.linspace(x_fit[0], x_fit[-1], 100)
    if extension:
        x_linsp=np.linspace(extension[0], extension[1], extension[2])
        
    y_linsp = model(x_linsp, *opt)
        
    b_opt = []
    b_c2r = []
    b_y_linsp = []
    
    for j in range(n_boot):
        y_fit_j = [b_y_fit[i][j] for i in range(len(b_y_fit))]
        
        opt_j, cov_j = curve_fit(model, x_fit, y_fit_j, sigma=d_y_fit, absolute_sigma=True)
        b_opt.append(opt_j)

        b_c2r.append(reg.chi2_corr(x_fit, y_fit_j, model, np.diag(np.array(d_y_fit)**2), *opt_j))
        
        y_linsp_j = model(x_linsp, *opt_j)
        b_y_linsp.append(y_linsp_j)

    d_opt = np.std(b_opt, axis=0, ddof=1)
    
    c2r = reg.chi2_corr(x_fit, y_fit, model, np.diag(np.array(d_y_fit)**2), *opt)
    d_c2r = np.std(b_c2r)
    
    d_y_linsp = np.std(b_y_linsp, axis=0, ddof=1)
    return x_linsp, y_linsp, d_y_linsp, opt, d_opt, b_opt, c2r, d_c2r              

## specific
def readfile(path):
    output = np.loadtxt(f'{path}/analysis/Wloop.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    return np.column_stack(columns)
    
def thermalization(path, wt_max, ws_max):
    os.makedirs(f'{path}/analysis/therm/', exist_ok=True)
                  
    for ws in range(0,ws_max, ws_max//2):
        for wt in range(0, wt_max, wt_max//2):
            output = np.loadtxt(f'{path}/analysis/Wloop.dat', skiprows=1)
            Wloop_wt_ws = output[:, wt + wt_max*ws]
    
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
            Wloop = data[:, wt + wt_max*ws]

            boot.blocksize_analysis_primary(Wloop, 500, [10, len(Wloop)//10, 20], savefig=1, path=f"{path}/analysis/bsa", extra=f'wt{wt}_ws{ws}')
            
def get_potential (path, wt_max, ws_max):

    data = readfile(path)
    def id(x):
        return x

    wsplot = np.arange(1,ws_max+1)

    seed = 8220
    samples = 500
    blocksizes = [2000] * len(wsplot) 
                
    V = []
    d_V = []
    V_bs = []
    
    for ws, blocksize in zip(range(ws_max), blocksizes):
        V_ws = []
        d_V_ws = []
        V_bs_ws = []
        
        for wt in range(wt_max):
            W = data[:, wt + wt_max*ws]

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
    
    ## effective mass plot 
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$')

    for i in range(ws_max_plot):
        x_plot = np.arange(1,wt_max_plot[i] + 1)
        effm = res[i]
        d_effm = d_res[i]
        plt.errorbar(x_plot, effm[:wt_max_plot[i]], d_effm[:wt_max_plot[i]], **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
                
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend(loc='upper left')
    
    plt.savefig(f"{path}/analysis/effmass.png", dpi=300, bbox_inches='tight')
    
    ## effective mass plot vs 1/wt 
    plt.figure(figsize = (18,12))
    
    plt.xlabel(r'$1/w_t$')
    plt.ylabel(r'$aV(w_t)$')

    for i in range(ws_max_plot):
        x_plot = np.arange(1,wt_max_plot[i] + 1)
        effm = res[i]
        d_effm = d_res[i]
        plt.errorbar(1/x_plot, effm[:wt_max_plot[i]], d_effm[:wt_max_plot[i]], **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
                
    plt.xlim(0, 0.2)
    plt.ylim()
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend(loc='upper right')
    
    plt.savefig(f"{path}/analysis/pot_ws.png", dpi=300, bbox_inches='tight')

def find_wtmin(path, ws_max_plot, wt_start=1, wt_end=12):
    def model(x, a, b):
        return a + b/x
    
    data = np.load(f"{path}/analysis/potential_wt.npy", allow_pickle=True)
    
    res, d_res, boot_res = map(np.array, data[1:4])
    
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_{t,min}$')
    plt.ylabel(r'$aV(w_t=\infty)$')
    for i in range(ws_max_plot):
        dp = []
        d_dp = []
        for wt_min_fit in range(wt_start,wt_end-3):
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
            
        if i == 0:
            ylim_inf = dp[-1] * (1-0.2)
        if i==ws_max_plot-1:
            ylim_sup = dp[0] * (1 + 0.2)
         
        
        plt.errorbar(range(wt_start, wt_end-3), dp, d_dp, **plot.data(i), label=f'$w_s$={i+1}')
    
    plt.ylim(ylim_inf, ylim_sup)
    plt.legend(loc='upper left')
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"{path}/analysis/find_wtmin.png", dpi=300, bbox_inches='tight')    

def extrap_potential(path, wt_max_plot, ws_max_plot, wt_min_fit, wt_max_fit):
    def model(x, a, b):
        return a + b*x
    
    data = np.load(f"{path}/analysis/potential_wt.npy", allow_pickle=True)
    
    wsplot, res, d_res, boot_res = map(np.array, data[:4])
        
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$1/w_t$')
    plt.ylabel(r'$aV(w_t)$')
    
    pot, b_pot = [], []
            
    print("ws pot_n d_pot_n c2r_n d_c2r_b")
    for i in range(ws_max_plot):
        Mp = wt_max_plot[i]
        mf, Mf = wt_min_fit[i], wt_max_fit[i]
        x, y, d_y, b_y = 1/np.arange(1,Mp+1), np.array(res[i][0:Mp]), np.array(d_res[i][0:Mp]), np.array(boot_res[i][0:Mp])

        if i == 0:
            ylim_inf = y[-1]*0.5
        if i==ws_max_plot-1:
            ylim_sup = y[0]*0.8
        
        x_linsp, y_linsp, d_y_linsp, opt, d_opt, b_opt, c2r, d_c2r = boot_fit(x, y, d_y, b_y, model, mf, Mf, [0, 0.12, 2])
        
        pot.append(opt[0])
        b_pot.append([b_opt[j][0] for j in range(len(b_opt))])
        
        plt.errorbar(x, y, d_y, **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$, $\chi^2/dof$={c2r:.2f}, $m$={mf}, $M$={Mf} ")
        
        plt.plot(x_linsp, y_linsp, **plot.fit(i))
        plt.fill_between(x_linsp, y_linsp-d_y_linsp, y_linsp+d_y_linsp, **plot.conf_band(i))

        print(f"{wsplot[i]:.2g} {opt[0]:.6g} {d_opt[0]:.2g} {c2r:.4g} {d_c2r:.4g}")
        
    plt.xlim(-0.01, 0.12)
    plt.ylim((ylim_inf,ylim_sup))
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/analysis/extrap_pot.png", dpi=300, bbox_inches='tight')    

    wsplot = wsplot[:ws_max_plot].copy()
    save = np.array([wsplot, pot, b_pot], dtype=object)
    np.save(f"{path}/analysis/opt", save)       
  
## main
def planarpot(path, ws_max, ws_max_plot, wt_max, wt_min_fit, wt_max_fit):    
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
        find_wtmin(path, ws_max_plot, wt_start=2, wt_end=21)
    
    extrap_potential(path, wt_max_plot, ws_max_plot, wt_min_fit, wt_max_fit)
    
def potential(x, a, b, c):
    return a + b*x + c*np.log(x)

def plotfit_potential(path, fit='false', print_res='true', ri=[1, 5**0.5, np.sqrt(8)], start=0, end=10):   
    x, y, b_y = np.load(f"{path}/analysis/opt.npy", allow_pickle=True)
    
    d_y = [np.std(b_y[i], ddof=1) for i in range(len(b_y))]  

    plt.figure(figsize=(18,12))
    plt.errorbar(x, y, d_y, **plot.data(0))
    plt.xlabel(r"r/a")
    plt.ylabel(r"aV(r/a)")
    
    if(fit == 'true'):
        x_linsp, y_linsp, d_y_linsp, opt, d_opt, b_opt, c2r, d_c2r = boot_fit(x, y, d_y, b_y, potential, start, end)
           
        plt.plot(x_linsp, y_linsp, **plot.fit(4))
        plt.fill_between(x_linsp, y_linsp-d_y_linsp, y_linsp+d_y_linsp, **plot.conf_band(4))

        if print_res == 'true':
            print("opt")
            print(opt)
            print("d_opt")
            print(d_opt)
            print("c2r d_c2r")
            print(f'{c2r:.4f} {d_c2r:.4f}')
            
    plt.axvline(ri[0])
    plt.axvline(ri[1])
    plt.axvline(ri[2])

    plt.title(f'c2r = {c2r:.4f}\n opt = {opt}\n d_opt = {d_opt}')
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"{path}/analysis/potential_{int(ri[0]**2)}_{int(ri[1]**2)}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    np.save(f"{path}/analysis/opt_potential", np.array([opt, b_opt], dtype=object))

def compute_r2F(path, ri = [1, 5**0.5, np.sqrt(8)]):
    tmp = np.load(f"{path}/analysis/opt_potential.npy", allow_pickle=True)
    opt, b_opt = tmp[0], tmp[1]

    r1 = ri[0]
    r2 = ri[1]
    r3 = ri[2]
    
    y_r1 = potential(r1, *opt)
    y_r2 = potential(r2, *opt)
    y_r3 = potential(r3, *opt)

    b_r2F_r1, b_r2F_r2 = [], []
    
    for opt_j in b_opt:
        b_r2F_r1.append(r1**2*(potential(r2, *opt_j)-potential(r1, *opt_j))/(r2-r1))
        b_r2F_r2.append(r2**2*(potential(r3, *opt_j)-potential(r2, *opt_j))/(r3-r2))
    
    r2F_r1 = r1**2*(y_r2-y_r1)/(r2-r1)
    d_r2F_r1 = np.std(b_r2F_r1, ddof=1)
    r2F_r2 = r2**2*(y_r3-y_r2)/(r3-r2)
    d_r2F_r2 = np.std(b_r2F_r2, ddof=1)
    
    print('r2F_r1(d_r2F_r1)')
    format(r2F_r1, d_r2F_r1)
    print('r2F_r2(d_r2F_r2)')
    format(r2F_r2, d_r2F_r2)

    r2F = [r2F_r1, r2F_r2]
    b_r2F = [b_r2F_r1, b_r2F_r2]
    
    np.save(f"{path}/analysis/r2F_{int(ri[0]**2)}_{int(ri[1]**2)}", np.array([r2F, b_r2F], dtype=object))

## flag 0->r1 1->r2
def tune_r2F(beta_lst, rLi_lst, flag=0):
    
    def apbxpcx2(x,a,b,c):
        return a + b*x +c*x**2
    
    def zeros(x):
        return (apbxpcx2(x,*opt) - y_0)

    tuned_betas = []
    
    plt.figure(figsize=(18,12))
    for i, rLi in enumerate(rLi_lst):
        y_lst, d_y_lst, b_y_lst = [], [], []
        
        for beta in beta_lst[i]:
            path =  f"/home/negro/projects/matching/step_scaling/infvol_PBC/Nt42_Ns42_b{beta}"

            tmp = np.load(f"{path}/analysis/r2F_{int(rLi[0]**2)}_{int(rLi[1]**2)}.npy", allow_pickle=True)
            r2F, b_r2F = tmp[0], tmp[1]
            
            y_tmp, b_y_tmp = r2F[flag], b_r2F[flag]
                                                
            d_y_tmp = np.std(b_y_tmp, ddof=1) 
                    
            label=fr'$r_1$={rLi[0]:.2f}' if beta==beta_lst[i][0] else None
            plt.errorbar(beta, y_tmp, d_y_tmp, **plot.data(i), label=label)
                       
            y_lst.append(y_tmp)
            d_y_lst.append(d_y_tmp)
            b_y_lst.append(b_y_tmp)
        
            if i==0 and beta==beta_lst[i][0]:
                beta_0 = beta
                y_0 = y_tmp
        if i!=0:
            x_linsp, y_linsp, d_y_linsp, opt, d_opt, b_opt, c2r, d_c2r = boot_fit(x=beta_lst[i], y=y_lst, d_y=d_y_lst, b_y=b_y_lst, model=apbxpcx2, lim_inf=0, lim_sup=len(y_lst))
            plt.plot(x_linsp, y_linsp, **plot.fit(i), label = fr"$\chi^2/\nu$={c2r:.2f}")
            plt.fill_between(x_linsp, y_linsp-d_y_linsp, y_linsp+d_y_linsp, **plot.conf_band(i))
        
            ## finding which value of the bare coupling corresponds to the same physical coupling
            beta_tuned = fsolve(zeros, beta_0)
        
            tuned_betas.append(beta_tuned)
        
            x_tuned = np.array(tuned_betas)[i-1]
            y_tuned = apbxpcx2(x_tuned,*opt)
            b_y_tuned = [apbxpcx2(x_tuned, *opt_j) for opt_j in b_opt]
            d_y_tuned = np.std(b_y_tuned, ddof=1)
            plt.errorbar(x_tuned, y_tuned, d_y_tuned, **plot.data(0))
    
    plt.xlabel(fr'$\beta=1/g^2$')
    plt.ylabel(fr'$r_1^2F(r_{flag+1}/a,g)$')
    plt.legend()
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"/home/negro/projects/matching/step_scaling/infvol_PBC/r2F_r{flag+1}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return tuned_betas

def colim_vs_beta(beta_lst, rLi_lst, tuned_betas, flag=1):
    def axpbpcx2(x,a,b,c):
        return a + b*x + c*x**2
    
    res, d_res = [], []

    plt.figure(figsize=(18,12))  
    for i, rLi in enumerate(rLi_lst):
        y_lst, d_y_lst, b_y_lst = [], [], []
        
        for beta in beta_lst[i]:
            path =  f"/home/negro/projects/matching/step_scaling/infvol_PBC/Nt42_Ns42_b{beta}"

            tmp = np.load(f"{path}/analysis/r2F_{int(rLi[0]**2)}_{int(rLi[1]**2)}.npy", allow_pickle=True)
            r2F, b_r2F = tmp[0], tmp[1]
            
            y_tmp = r2F[flag]
            b_y_tmp = b_r2F[flag]
            d_y_tmp = np.std(b_y_tmp, ddof=1)
            
            label=fr'$r_2$={rLi[1]:.2f}' if beta==beta_lst[i][0] else None
            plt.errorbar(beta, y_tmp, d_y_tmp, **plot.data(i), label=label)

            y_lst.append(y_tmp)
            d_y_lst.append(d_y_tmp)
            b_y_lst.append(b_y_tmp)
        
            if i==0 and beta==beta_lst[i][0]:
                res.append(y_tmp)
                d_res.append(d_y_tmp)

        if i!=0:
            x_linsp, y_linsp, d_y_linsp, opt, d_opt, b_opt, c2r, d_c2r = boot_fit(x=beta_lst[i], y=y_lst, d_y=d_y_lst, b_y=b_y_lst, model=axpbpcx2, lim_inf=0, lim_sup=len(y_lst))
            
            plt.plot(x_linsp, y_linsp, **plot.fit(i), label = fr"$\chi^2/\nu$={c2r:.2f}")
            plt.fill_between(x_linsp, y_linsp-d_y_linsp, y_linsp+d_y_linsp, **plot.conf_band(i))
            
            x_tuned = tuned_betas[i-1]
            y_tuned = axpbpcx2(x_tuned,*opt)
            b_y_tuned = [axpbpcx2(x_tuned, *opt_j) for opt_j in b_opt]
            d_y_tuned = np.std(b_y_tuned, ddof=1)
            plt.errorbar(x_tuned, y_tuned, d_y_tuned, **plot.data(0))
            
            res.append(np.float64(y_tuned[0]))
            d_res.append(np.float64(d_y_tuned))
    
    plt.ylabel(fr'$r_2^2F(r_{flag+1}/a,g)$')
    plt.xlabel(fr'$\beta=1/g^2$')
    plt.legend(loc='lower left')
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"/home/negro/projects/matching/step_scaling/infvol_PBC/r2F_r{flag+1}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return res, d_res
     
def colim_vs_r2latt(r_latt, res, d_res):
    x_fit = np.array(1/r_latt**2)
    
    plt.figure(figsize=(18,12))
    
    plt.errorbar(x_fit, res, d_res, **plot.data(0))
    
    plt.xlabel(r'$1/r^2_{latt}$')
    plt.ylabel(r'$\bar{g}(r)$')
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"/home/negro/projects/matching/step_scaling/infvol_PBC/continuum_limit.png", dpi=300, bbox_inches='tight')
    plt.close()
 
## main functions      
 
def firstage():
    path_glob = f"/home/negro/projects/matching/step_scaling/infvol_PBC/Nt42_Ns42_b3.25"
    
    if not os.path.isdir(f"{path_glob}/analysis/"):
        os.makedirs(f"{path_glob}/analysis/") 
        
    if not os.path.isfile(f"{path_glob}/analysis/Wloop.dat"):
        concatenate.concatenate(f"{path_glob}/data", 500, f"{path_glob}/analysis/")
    
    ## planar
    wt_max_plot = 21
    ws_max = 7
    ws_max_plot = 7
    wt_min_fit = [16] * ws_max_plot
    wt_max_fit = [21] * ws_max_plot
    
    planarpot(path_glob, ws_max, ws_max_plot, wt_max_plot, wt_min_fit, wt_max_fit)
    
    rL3 = [1, 5**0.5, 8**0.5]
    rL4 = [2**0.5, 10**0.5, 18**0.5]
    rL5 = [5**0.5, 25**0.5, 40**0.5]
    rL7 = [8**0.5, 40**0.5, 72**0.5]
    
    rL = [rL5]
        
    for rLi in rL:
         plotfit_potential(path_glob, fit='true', ri=rLi, start=1, end=7)
         compute_r2F(path_glob, ri=rLi)
    
def secondstage():
    ## list of betas available
    beta_0 = [2]  
    beta_lst_1 = [2.25, 2.26, 2.27, 2.28, 2.29, 2.3, 2.5] ## analyze 2.35 2.4 2.45
    beta_lst_2 = [2.8, 2.85, 2.9, 2.95, 3, 3.05, 3.1, 3.15, 3.2]
    beta_lst_3 = [3.25, 4, 5] # 3.5 3.75
    
    beta_lst = [beta_0, beta_lst_1, beta_lst_2, beta_lst_3]
    
    ## different distances used
    rL3 = [1, 5**0.5, 8**0.5]
    rL4 = [2**0.5, 10**0.5, 18**0.5]
    rL5 = [5**0.5, 25**0.5, 40**0.5]
    rL7 = [8**0.5, 40**0.5, 72**0.5]
    
    ## distances to be used at the present moment
    rLi_lst = [rL3, rL4, rL5, rL7]
    
    ## flag=0 -> r1, flag=1 -> r2 
    tuned_betas = tune_r2F(beta_lst, rLi_lst, flag=0)
    
    res, b_res = colim_vs_beta(beta_lst, rLi_lst, tuned_betas, flag=1)
    
    r_latt = np.array([5**0.5, 10**0.5, 25**0.5, 40**0.5])
    colim_vs_r2latt(r_latt, res, b_res)
     
if __name__ == "__main__":

    #firstage()
    
    secondstage()

    