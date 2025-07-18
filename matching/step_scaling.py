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

def readfile(path, filename):
    output = np.loadtxt(f'{path}/analysis/{filename}.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    ## dati.dat
    # plaqs plaqt 
    ## Wloop.dat
    # W(wt=1,ws=1) ... W(wt=wtmax,ws=1) ... W(wt=1,ws=wsmax) ... W(wt=wtmax,ws=wsmax)
    ## sWloop.dat
    # W(wt=1,ws=1) ... W(wt=wtmax,ws=1) ... W(wt=1,ws=wsmax) ... W(wt=wtmax,ws=wsmax)

    return np.column_stack(columns)
    
def thermalization(path, wt_max, ws_max, type='planar'):
    os.makedirs(f'{path}/analysis/therm_{type}/', exist_ok=True)
    if type == 'planar':           
        filename = 'Wloop'           
    elif type == 'staircase':
        filename = 'sWloop' 
                  
    for ws in range(ws_max):
        for wt in range(wt_max//3):
            output = np.loadtxt(f'{path}/data/{filename}_0.dat', skiprows=1)
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

            plt.savefig(f"{path}/analysis/therm_{type}/therm_{filename}_wt{wt}_ws{ws}.png", bbox_inches='tight')
            plt.close()
        
def bsa_potential(path, wt_max, ws_max, type='planar'):
    os.makedirs(f'{path}/analysis/bsa_{type}/', exist_ok=True)
    seed = 8220

    if type == 'planar':           
        filename = 'Wloop'           
    elif type == 'staircase':
        filename = 'sWloop'
        
    data = readfile(path, filename)
    
    def id(x):
        return x
    
    for ws in range(ws_max):
        for wt in range(wt_max//3):
            Wloop = data[:, wt + wt_max*ws]

            args = [id, Wloop]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
    
            args = [id, Wloop]

            # start stop step
            block_vec = [2, len(Wloop)//100, 2]

            block_range = range(block_vec[0], block_vec[1], block_vec[2])

            err = []
    
            samples_boot = 200
            for block in block_range:                        
                _, bar = boot.bootstrap_for_secondary(potential, block, samples_boot, 0, args, seed=seed)
                
                err.append(bar)
        
            plt.figure(figsize=(18,12))
            plt.plot(block_range, err
                    , marker = 'o'
                    , linestyle = '-', linewidth = 0.375
                    , markersize = 2
                    , color = plot.color_dict[1])
                
            plt.xlabel(r'$K$')
            plt.ylabel(r'$\overline{\sigma}_{\overline{F^{(K)}}}$', rotation=0)
            plt.title("Standard error for secondary observables")

            plt.grid(True, which='both', linestyle='--', linewidth=0.25)
            plt.savefig(f"{path}/analysis/bsa_{type}/bsa_{filename}_wt{wt}_ws{ws}.png", bbox_inches='tight')
            plt.close()
            
def get_potential (path, wt_max, ws_max, wsplot=None, type='planar'):
    if type == 'planar':
        filename = 'Wloop'
    else:
        filename = 'sWloop'
        
    data = readfile(path, filename=filename)
    def id(x):
        return x
    
    if type == 'planar':
        wsplot = np.arange(1,ws_max+1)
    elif type == 'staircase':
        wsplot = wsplot
        
    seed = 8220
    samples = 1000
    blocksizes = [20] * len(wsplot) 
                
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
        
        
    np.save(f"{path}/analysis/{type}_potential_wt", np.array([wsplot, V, d_V, V_bs], dtype=object))
            
def plot_potential(path, wt_max_plot, ws_max_plot, type='planar'):
    data = np.load(f"{path}/analysis/{type}_potential_wt.npy", allow_pickle=True)
    
    wsplot, res, d_res, res_bs = map(np.array, data[:4])
            
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
    
    plt.savefig(f"{path}/analysis/{type}_effmass.png", dpi=300, bbox_inches='tight')
    
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
    
    plt.xlim(-0.01, 0.6)
    
    plt.savefig(f"{path}/analysis/{type}pot_ws.png", dpi=300, bbox_inches='tight')

def find_wtmin(path, ws_max_plot, wt_start, wt_end, type='planar'):
    def model(x, a, b):
        return a + b/x
    
    data = np.load(f"{path}/analysis/{type}_potential_wt.npy", allow_pickle=True)
    
    res, d_res, boot_res = map(np.array, data[1:4])
    
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_{t,min}$')
    plt.ylabel(r'$aV(w_t=\infty)$')
    for i in range(ws_max_plot):
        dp = []
        d_dp = []
        for wt_min_fit in range(wt_start,wt_end-4):
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
        
        plt.errorbar(range(wt_start, wt_end-4), dp, d_dp, **plot.data(i), label=i)
           
    plt.legend
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"{path}/analysis/find_wtmin_{type}.png", dpi=300, bbox_inches='tight')    

def extrap_potential(path, wt_max_plot, ws_max_plot, wt_min_fit, wt_max_fit, type='planar'):
    def model(x, a, b):
        return a + b/x
    
    data = np.load(f"{path}/analysis/{type}_potential_wt.npy", allow_pickle=True)
    
    wsplot, res, d_res, boot_res = map(np.array, data[:4])
        
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$')
    
    
    bounds = -np.inf, np.inf
    p0 = [1, 1]
    
    pot, boot_pot_glob = [], []
    
    # fit on the original sample 
    
    print("ws pot_n pot_b d_pot_n d_pot_b cov_b c2r_n c2r_b d_c2r_b")
    
    for i in range(ws_max_plot):
        Mp = wt_max_plot[i]
        mf, Mf = wt_min_fit[i], wt_max_fit[i]
        
        x_plot, y_plot, d_y_plot = np.arange(0,Mp), np.array(res[i][0:Mp]), np.array(d_res[i][0:Mp])
        x_fit, y_fit, d_y_fit = np.arange(mf,Mf), np.array(res[i][mf:Mf]), np.array(d_res[i][mf:Mf])
                
        opt, cov = curve_fit(model, x_fit, y_fit, sigma=d_y_fit, absolute_sigma=True, p0=p0, bounds=bounds)
        c2r = reg.chi2_corr(x_fit, y_fit, model, np.diag(d_y_fit**2), opt[0], opt[1])/(len(x_fit) - len(opt))
        
        pot.append(opt[0])
        
        x_linsp_fit = np.linspace(x_fit[0], x_fit[-1], 200)
        y_linsp_fit = model(x_linsp_fit, opt[0], opt[1])
        
        plt.errorbar(x_plot, y_plot, d_y_plot, **plot.data(i),
                     label=fr"$w_s={wsplot[i]:.2f}$, $\chi^2/dof$={c2r:.2f}")
        plt.plot(x_linsp_fit, y_linsp_fit, **plot.fit(i))
        
        p0 = opt

        boot_y_fit = np.array(boot_res[i][mf:Mf])
        n_boot = np.shape(boot_y_fit)[1]
        
        boot_pot, boot_cov, boot_band, boot_c2r = [], [], [], []

        for j in range(n_boot):
            y_fit = boot_y_fit[:,j]
            
            opt_j, cov_j = curve_fit(model, x_fit, y_fit, sigma=d_y_fit, absolute_sigma=True, p0=p0, bounds=bounds)
            
            c2r_j = reg.chi2_corr(x_fit, y_fit, model, np.diag(d_y_fit**2), opt_j[0], opt_j[1])/(len(x_fit) - len(opt_j))
            
            boot_pot.append(opt_j[0])
            boot_cov.append(cov_j[0][0]**0.5)
            boot_c2r.append(c2r_j)
            boot_band.append(model(x_linsp_fit, *opt_j))
        
        boot_band = np.std(boot_band, axis=0, ddof=1)
        plt.fill_between(x_linsp_fit, y_linsp_fit-boot_band, y_linsp_fit+boot_band, **plot.conf_band(i))
        
        boot_pot_glob.append(boot_pot)
        
        print(f"{wsplot[i]:.2g} {opt[0]:.6g} {np.mean(boot_pot):.6g} {cov[0][0]**0.5:.2g} {np.std(boot_pot, ddof=1):.2g} {np.mean(boot_cov):.2g} {c2r:.4g} {np.mean(boot_c2r):.4g} {np.std(boot_c2r, ddof=1):.2g}")    

    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/extrap_{type}pot.png", dpi=300, bbox_inches='tight')    

    wsplot = wsplot[:ws_max_plot].copy()
    save = np.array([wsplot, pot, boot_pot_glob], dtype=object)
    np.save(f"{path}/analysis/opt_{type}", save)

def compute_r2F(path, Ns=3):    
    planar = np.load(f"{path}/analysis/opt_planar.npy", allow_pickle=True)
    staircase = np.load(f"{path}/analysis/opt_staircase.npy", allow_pickle=True)
    
    ws, pot, boot_pot = [], [], []
    
    print(staircase[0])
    
    if Ns == 3:
        ws.extend([planar[0][0], staircase[0][1], staircase[0][-1]])
        pot.extend([planar[1][0], staircase[1][1], staircase[1][-1]])
        boot_pot.extend([planar[2][0], staircase[2][1], staircase[2][-1]])
    if Ns == 4:
        ws.extend([staircase[0][0], staircase[0][3], staircase[0][-1]])
        pot.extend([staircase[1][0], staircase[1][3], staircase[1][-1]])
        boot_pot.extend([staircase[2][0], staircase[2][3], staircase[2][-1]])
    if Ns == 5:
        ws.extend([staircase[0][1], staircase[0][8], staircase[0][-1]])
        pot.extend([staircase[1][1], staircase[1][8], staircase[1][-1]])
        boot_pot.extend([staircase[2][1], staircase[2][8], staircase[2][-1]])
    if Ns == 7:
        ws.extend([staircase[0][2], staircase[0][15], staircase[0][-1]])
        pot.extend([staircase[1][2], staircase[1][15], staircase[1][-1]])
        boot_pot.extend([staircase[2][2], staircase[2][15], staircase[2][-1]])
        
        
    r2F_r1 = ws[0]**2*(pot[1]-pot[0])/(ws[1]-ws[0])
    r2F_r2 = ws[1]**2*(pot[2]-pot[1])/(ws[2]-ws[1])
    
    boot_r2F_r1 = ws[0]**2*(np.array(boot_pot[1])-np.array(boot_pot[0]))/(ws[1]-ws[0])
    boot_r2F_r2 = ws[1]**2*(np.array(boot_pot[2])-np.array(boot_pot[1]))/(ws[2]-ws[1])
    
    data = np.array([[ws[0], ws[1]], [r2F_r1, r2F_r2], [boot_r2F_r1, boot_r2F_r2]], dtype=object)
    
    d_r2F_r1 = np.std(boot_r2F_r1, ddof=1)
    d_r2F_r2 = np.std(boot_r2F_r2, ddof=1)

    print("r1 r2 r2F_r1 d_r2F_r1 r2F_r2 d_r2F_r2")
    print(ws[0], ws[1], r2F_r1, d_r2F_r1, r2F_r2, d_r2F_r2)
    
    np.save(f"{path}/analysis/r2F", data)
    
def fit_staircase(path, lim_inf):
    staircase = np.load(f"{path}/analysis/opt_staircase.npy", allow_pickle=True)
    
    x = staircase[0]
    y = staircase[1]
    b_y = staircase[2]
    d_y = [np.std(b_y[j], ddof=1) for j in range(len(b_y))]    
    
    def luscher_pot(x, a, b, c):
        return a + b*x + c/x
    def log_pot(x, a, b, c):
        return a + b*x + c*np.log(x)
    def linear_pot(x, a, b):
        return a + b*x

    x_linsp_log, y_linsp_log, d_y_linsp_log, opt_log, d_opt_log, c2r_log, d_c2r_log = boot_fit(x, y, d_y, b_y, model=log_pot, lim_inf=lim_inf)
    x_linsp_lusc, y_linsp_lusc, d_y_linsp_lusc, opt_lusc, d_opt_lusc, c2r_lusc, d_c2r_lusc = boot_fit(x, y, d_y, b_y, model=luscher_pot, lim_inf=lim_inf)
    x_linsp_lin, y_linsp_lin, d_y_linsp_lin, opt_lin, d_opt_lin, c2r_lin, d_c2r_lin = boot_fit(x, y, d_y, b_y, model=linear_pot, lim_inf=lim_inf)

    plt.figure(figsize=(18,12))
    
    plt.errorbar(x, y, d_y, **plot.data(0))#
    
    # plt.plot(x_linsp_log, y_linsp_log, **plot.fit(1), label=fr"a+$\sigma$*x+c*ln(x), $\sigma$={opt_log[1]:.4g}({d_opt_log[1]:.2g}), $\chi2/\nu$={c2r_log:.2g}({d_c2r_log:.2g})")
    # plt.fill_between(x_linsp_log, y_linsp_log-d_y_linsp_log,y_linsp_log+d_y_linsp_log, **plot.conf_band(1))
    
    # plt.plot(x_linsp_lusc, y_linsp_lusc, **plot.fit(2), label=fr"a+$\sigma$*x+c/x, $\sigma$={opt_lusc[1]:.4g}({d_opt_lusc[1]:.2g}), $\chi2/\nu$={c2r_lusc:.2g}({d_c2r_lusc:.2g})")
    # plt.fill_between(x_linsp_lusc, y_linsp_lusc-d_y_linsp_lusc,y_linsp_lusc+d_y_linsp_lusc, **plot.conf_band(2))
    
    plt.plot(x_linsp_lin, y_linsp_lin, **plot.fit(3), label=fr"a+$\sigma$*x, $\sigma$={opt_lin[1]:.4g}({d_opt_lin[1]:.2g}), $\chi2/\nu$={c2r_lin:.2g}({d_c2r_lin:.2g})")
    plt.fill_between(x_linsp_lin, y_linsp_lin-d_y_linsp_lin,y_linsp_lin+d_y_linsp_lin, **plot.conf_band(3))
    
    plt.legend()    
    
    plt.savefig(f"{path}/analysis/fit_staircase.png", dpi=300, bbox_inches='tight')     
    
def boot_fit(x, y, d_y, b_y, model, lim_inf):
    x_fit, y_fit, d_y_fit, b_y_fit = x[lim_inf:], y[lim_inf:], d_y[lim_inf:], b_y[lim_inf:]
    opt, cov = curve_fit(model, x_fit, y_fit, sigma=d_y_fit, absolute_sigma=True)
    
    n_boot = len(b_y[0])

    x_linsp = np.linspace(x[0], x[-1], 100)
    y_linsp = model(x_linsp, *opt)
    
    b_opt_log = []
    b_c2r = []
    b_y_linsp = []
    
    for j in range(n_boot):
        y_fit_j = [b_y_fit[i][j] for i in range(len(b_y_fit))] 
        
        opt_j, cov_j = curve_fit(model, x_fit, y_fit_j, sigma=d_y_fit, absolute_sigma=True)
        
        b_c2r.append(reg.chi2_corr(x_fit, y_fit_j, model, np.diag(np.array(d_y_fit)**2), *opt_j))
        
        b_opt_log.append(opt_j)
        
        y_linsp_j = model(x_linsp, *opt_j)
        
        b_y_linsp.append(y_linsp_j)
    
    
    b_opt = b_opt_log
    d_opt = np.std(b_opt_log, axis=0, ddof=1)
    
    c2r = reg.chi2_corr(x_fit, y_fit, model, np.diag(np.array(d_y_fit)**2), *opt)
    d_c2r = np.std(b_c2r, ddof=1)
    
    d_y_linsp = np.std(b_y_linsp, axis=0, ddof=1)
    
    return x_linsp, y_linsp, d_y_linsp, opt, d_opt, c2r, d_c2r       
 
def get_r2F_old(path):
    tmp = np.loadtxt(f'{path}/analysis/r2F.txt')
    
    x = tmp[:,0]
    y = tmp[:,1]
    d_y = tmp[:,2]
    
    return x, y, d_y
 
def plot_r2F_old(path, Nss, betas):
     
    plt.figure(figsize=(18,12))
    for i, Ns in enumerate(Nss):
        for beta in betas[i]:
            local_path = f'{path}/L{Ns}/T48_L{Ns}_b{beta}'
            x, y, d_y = get_r2F_old(local_path)
            
            plt.errorbar(beta, y[0], d_y[0], **plot.data(i))
    plt.savefig(f'{path}/tuning_b3/invtuning_b3_r1.png', dpi=300, bbox_inches='tight')
    plt.close()

## main

def planarpot(path):
    type = 'planar'
    
    wt_max = 48
    ws_max = 6
    
    if not os.path.isdir(f"{path}/analysis/therm_{type}/"):
        thermalization(path, wt_max, ws_max, type = type)
        
    if not os.path.isdir(f"{path}/analysis/bsa_{type}/"):
        bsa_potential(path, wt_max, ws_max, type = type)
    
    ws_max_plot = 6
    wt_max_plot = [40]*ws_max_plot
    
    if not os.path.isfile(f"{path}/analysis/planar_potential_wt.npy"):
         get_potential(path, wt_max, ws_max, type=type)
    
    plot_potential(path, wt_max_plot, ws_max_plot, type=type)

    wt_min_fit = [6] * ws_max_plot
    wt_max_fit = [40]*ws_max_plot
    
    if not os.path.isfile(f"{path}/analysis/find_wtmin_{type}.png"):
        find_wtmin(path, ws_max_plot, 2, 24, type='planar')
        
    extrap_potential(path, wt_max_plot, ws_max_plot, wt_min_fit, wt_max_fit, type=type)

def staircase(path, wsplot):    
    type = 'staircase'
    
    wt_max = 48
    ws_max = 21 # 21 for Ns 7 10 for Ns 5 
    
    if not os.path.isdir(f"{path}/analysis/therm_{type}/"):
        thermalization(path, wt_max, ws_max, type = type)
        
    if not os.path.isdir(f"{path}/analysis/bsa_{type}/"):
        bsa_potential(path, wt_max, ws_max, type = type)
    
    ws_max_plot = 21
    wt_max_plot = [40] * ws_max
    
    if not os.path.isfile(f"{path}/analysis/staircase_potential_wt.npy"):
        get_potential(path, wt_max, ws_max, wsplot=wsplot, type=type)
    
    plot_potential(path, wt_max_plot, ws_max_plot, type=type)

    wt_min_fit = [16] * ws_max_plot
    wt_max_fit = [24]* ws_max_plot
    
    if not os.path.isfile(f"{path}/analysis/find_wtmin_{type}.png"):
        find_wtmin(path, ws_max_plot, 2, 24, type='staircase')
    extrap_potential(path, wt_max_plot, ws_max_plot, wt_min_fit, wt_max_fit, type=type)

def plotfit_potential(path, fit = 'false'):   
    px, py, bpy = np.load(f"{path}/analysis/opt_planar.npy", allow_pickle=True)
    sx, sy, bsy = np.load(f"{path}/analysis/opt_staircase.npy", allow_pickle=True)
      
    d_py = [np.std(bpy[i], ddof=1) for i in range(len(bpy))]  
    d_sy = [np.std(bsy[i], ddof=1) for i in range(len(bsy))]  

    plt.figure(figsize=(18,12))
    plt.errorbar(px, py, d_py, **plot.data(1))
    plt.errorbar(sx, sy, d_sy, **plot.data(2))
    plt.xlabel(r"r/a")
    plt.ylabel(r"aV(r/a)")
    
    if(fit == 'true'):
        start = 0
        
        px, sx, py, sy, d_py, d_sy, bpy, bsy = px[start:], sx[start:], py[start:], sy[start:], d_py[start:], d_sy[start:], bpy[start:], bsy[start:]
        
        x, y, d_y = np.concatenate([px, sx]), np.concatenate([py, sy]), np.concatenate([d_py, d_sy])
        
        def potential(x, a, b, c):
            return a + b*x + c*np.log(x)
            
        opt, cov = curve_fit(potential, x, y, sigma = d_y, absolute_sigma=True)
        
        c2r = reg.chi2_corr(x, y, potential, np.diag(d_y**2), *opt)/(len(x) - len(opt))
        
        x_fit = np.linspace(min(x), max(x), 100)
        
        by = np.concatenate([bpy, bsy])
        
        boot_opt, boot_band = [], []
        
        for j in range(len(by[0])):
            y_j = np.array([by[i][j] for i in range(len(by))])
            opt_j, _ = curve_fit(potential, x, y_j, sigma=d_y, absolute_sigma=True)
            boot_opt.append(opt_j)
            boot_band.append(potential(x_fit, *opt_j))
            
        boot_band = np.std(boot_band, axis=0, ddof=1)

        boot_opt = np.array(boot_opt)
        opt_err = np.std(boot_opt, axis=0, ddof=1)
        
        y_fit = potential(x_fit, *opt)
        
        plt.plot(x_fit, y_fit, **plot.fit(4), label=fr"$\sigma$={opt[1]:.4g}({opt_err[1]:.4g})")
        plt.fill_between(x_fit, y_fit-boot_band, y_fit+boot_band, **plot.conf_band(4))
        
        print('x')
        print(x)
        print('y')
        print(np.array(y))
        print('d_y')
        print(d_y)

        print('sigma d_sigma d_sigma_boot c2r')
        print(f'{opt[1]:.6g} {cov[1][1]**0.5:.4g} {opt_err[1]:.4g} {c2r:.4g}')    
        
        plt.legend()

    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.savefig(f"{path}/analysis/potential.png", dpi=300, bbox_inches='tight')    

if __name__ == "__main__":
    Ns = 7
    beta = 12
    
    path_glob = f"/home/negro/projects/matching/step_scaling/CLSS/Nt48_Ns{Ns}/b{beta}"
    
    if not os.path.isfile(f"{path_glob}/analysis/Wloop.dat"):
            if not os.path.isfile(f"{path_glob}/analysis/sWloop.dat"):
                concatenate.concatenate(f"{path_glob}/data", 2000, f"{path_glob}/analysis")
    
    # ## planar
    # planarpot(path_glob)

    # ## staircase
    wsplot = np.array([np.sqrt(i**2 + j**2) for i in range(1,Ns) for j in range(1,i+1)])

    #staircase(path_glob, wsplot=wsplot)

    #plotfit_potential(path_glob)
    
    # compute_r2F(path_glob, Ns=Ns)
    
    fit_staircase(path_glob, lim_inf=5)
    
    path_root = f'/home/negro/projects/matching/step_scaling'
    Nss = [3, 4, 5, 7]
    betas = [[3], [4, 4.05, 4.1, 4.15, 4.2, 4.25], [11, 11.5, 12, 12.5], [10, 11, 12]]
    plot_r2F_old(path_root, Nss, betas)