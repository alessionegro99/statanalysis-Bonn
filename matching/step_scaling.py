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
    
# def tune_r2F():
#     Ns_list=[4, 5]
    
#     stuff = []
#     runcoup_r1 = []
#     d_runcoup_r1 = []
    
#     betas_list = [[4, 4.05, 4.1, 4.15, 4.2, 4.25], [11, 11.5, 12, 12.5]]

#     plt.figure(figsize=(18,12))

#     r = []
#     r2F=[]
#     d_r2F=[]
    
#     path = f"/home/negro/projects/matching/step_scaling/L3/T48_L3_b3"
    
#     ## r = r2
#     tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
#     r.append(tmp1)
#     r2F.append(tmp2)
#     d_r2F.append(tmp3)
    
#     r = np.column_stack(r)
#     r2F = np.column_stack(r2F)
#     d_r2F = np.column_stack(d_r2F)
    
#     r2F_L3 = r2F[:,:]
    
#     plt.errorbar(3, r2F[1, :], d_r2F[1,:], **plot.data(0))
    
#     for count, Ns in enumerate(Ns_list):
#         r = []
#         r2F=[]
#         d_r2F=[]
                
#         betas = betas_list[count]
#         for beta in betas:
#             path = f"/home/negro/projects/matching/step_scaling/L{Ns}/T48_L{Ns}_b{beta}"
            
#             tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
#             r.append(tmp1)
#             r2F.append(tmp2)
#             d_r2F.append(tmp3)
    
#         r = np.column_stack(r)
#         r2F = np.column_stack(r2F)
#         d_r2F = np.column_stack(d_r2F)
        
#         plt.errorbar(betas, r2F[1, :], d_r2F[1, :], **plot.data(count+1))
        
#         ## fitting 
#         # quadratic model
#         def model(x, a, b, c):
#             return a + b*x +c*x**2
        
#         # data to fit
#         x = betas
#         y = r2F[1, :]
#         d_y = d_r2F[1, :]
        
#         # starting parameters and bounds
#         p0 = [1,1,1]
#         bounds =  -np.inf, np.inf

#         # fitting
#         opt, cov = curve_fit(model, betas, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
#         # x_fit & y_fit
#         x_fit = np.linspace(x[0], x[-1], 100)
#         y_fit = model(x_fit, opt[0], opt[1], opt[2])

#         # plotting the fit curve
#         plt.plot(x_fit, y_fit, **plot.fit(count+1), label = 'null')
        
#         ## confidence band
#         # bootstrap samples
#         n_boot = 200

#         boot_opt = np.random.multivariate_normal(opt, cov, size=n_boot, tol=1e-10)
        
#         boot_y_fit = [model(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]        
#         boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
#         plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(count + 1))
        
#         ## getting tuned point and error
#         def func(x):
#             return (model(x, opt[0], opt[1], opt[2]) - r2F_L3[1,:])
#         beta0 = 4
#         beta_tuned = fsolve(func, beta0)
        
#         stuff.append(beta_tuned)
        
#         r2F_tuned = model(beta_tuned, opt[0], opt[1], opt[2])
#         boot_r2F_tuned = [model(beta_tuned, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]
#         d_r2F_tuned = np.std(boot_r2F_tuned, axis = 0, ddof=1)
        
#         plt.errorbar(beta_tuned, r2F_tuned, d_r2F_tuned, **plot.data(0))
        
#     ## r = r1
#     r = []
#     r2F=[]
#     d_r2F=[]
    
#     tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
#     r.append(tmp1)
#     r2F.append(tmp2)
#     d_r2F.append(tmp3)
    
#     r = np.column_stack(r)
#     r2F = np.column_stack(r2F)
#     d_r2F = np.column_stack(d_r2F)
    
#     plt.errorbar(3, r2F[0, :], d_r2F[0,:], **plot.data(0))
    
#     r2F_L3 = r2F[:,:]

#     for count, Ns in enumerate(Ns_list):
#         r = []
#         r2F=[]
#         d_r2F=[]
                
#         betas = betas_list[count]
#         for beta in betas:
#             path = f"/home/negro/projects/matching/step_scaling/L{Ns}/T42_L{Ns}_b{beta}"
            
#             tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
#             r.append(tmp1)
#             r2F.append(tmp2)
#             d_r2F.append(tmp3)
    
#         r = np.column_stack(r)
#         r2F = np.column_stack(r2F)
#         d_r2F = np.column_stack(d_r2F)
        
#         plt.errorbar(betas, r2F[0, :], d_r2F[0, :], **plot.data(count+1))
        
#         ## fitting 
#         # quadratic model
#         def model(x, a, b, c):
#             return a + b*x +c*x**2
        
#         # data to fit
#         x = betas
#         y = r2F[0, :]
#         d_y = d_r2F[0, :]
        
#         # starting parameters and bounds
#         p0 = [1,1,1]
#         bounds =  -np.inf, np.inf

#         # fitting
#         opt, cov = curve_fit(model, betas, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
#         # x_fit & y_fit
#         x_fit = np.linspace(x[0], x[-1], 100)
#         y_fit = model(x_fit, opt[0], opt[1], opt[2])

#         # plotting the fit curve
#         plt.plot(x_fit, y_fit, **plot.fit(count+1), label = 'null')
        
#         ## confidence band
#         # bootstrap samples
#         n_boot = 200

#         boot_opt = np.random.multivariate_normal(opt, cov, size=n_boot, tol=1e-10)
        
#         boot_y_fit = [model(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]        
#         boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
#         plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(count + 1))
        
#         ## getting tuned point and error
#         def func(x):
#             return (model(x, opt[0], opt[1], opt[2]) - r2F_L3[0,:])
#         beta_tuned = stuff[count]
        
#         r2F_tuned = model(beta_tuned, opt[0], opt[1], opt[2])
#         boot_r2F_tuned = [model(beta_tuned, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]
#         d_r2F_tuned = np.std(boot_r2F_tuned, axis = 0, ddof=1)
        
#         runcoup_r1.append(r2F_tuned)
#         d_runcoup_r1.append(d_r2F_tuned)
        
#         plt.errorbar(beta_tuned, r2F_tuned, d_r2F_tuned, **plot.data(0))
        
#     plt.grid (True, linestyle = '--', linewidth = 0.25)

#     plt.xlabel(r"$\beta$")
#     plt.ylabel(r"$r^2F(r,g)$", rotation=0)
#     #plt.savefig("/home/negro/projects/matching/step_scaling/r2F_tune_plot.png", dpi=300, bbox_inches='tight')
#     plt.show()
#     tofile = np.column_stack((np.array(stuff), np.array(runcoup_r1), np.array(d_runcoup_r1)))

#     np.savetxt("/home/negro/projects/matching/step_scaling/r2F_tune.txt", tofile)
    
# def plot_r2F_vs_rlatt(path):
    
#     r_latt = [1.00, 2**0.5, 5**0.5]
    
#     r2F = []
#     d_r2F = []
    
#     # L=3
#     L3_path = f"/home/negro/projects/matching/step_scaling/L3/T42_L3_b3"
#     tmp1, tmp2 = np.loadtxt(f"{L3_path}/analysis/r2F.txt", usecols=(1,2), unpack=True)

#     print(tmp1[0], tmp2[0])
    
#     r2F.append(tmp1[0])
#     d_r2F.append(tmp2[0])

#     ## L\in{4,5}
#     plt.figure(figsize = (16,12))
    
#     plt.xlabel(r'$1/r^2_{latt}$')
#     plt.ylabel(r'$r^2F(r_1,g)$', rotation = 0)

#     tmp1, tmp2 = np.loadtxt(f"{path}/r2F_tune.txt", usecols=(1,2), unpack=True)
    
#     r2F.append(tmp1[0])
#     r2F.append(tmp1[1])

#     d_r2F.append(tmp2[0])
#     d_r2F.append(tmp2[1])
    
#     r_latt = np.array(r_latt)
#     r2F = np.array(r2F)
#     d_r2F = np.array(d_r2F)
    
#     plt.errorbar(1/r_latt**2, r2F, d_r2F, **plot.data(1))
#     plt.gca().yaxis.set_label_coords(-0.1, 0.45)
    
#     plt.grid (True, linestyle = '--', linewidth = 0.25)

    
#     plt.savefig(f'{path}/r2F_r1.png', dpi=300, bbox_inches='tight')
#     plt.show()

# def confronto_r2F_r1_L4(Ns):
    
#     # [l=1, l=2, l=3]
#     x_ac = [1/0.36915425472851615**2, 1/0.7531542547285165**2, 1/0.7400542547285165**2]
#     r2F_r1_ac = [0.14874300045572017, 0.2400338548299101, 0.20420630542747306]
    
#     betas = [1.6, 1.8, 2, 4, 4.05, 4.15, 4.25]

#     plt.figure(figsize=(16,12))

#     r = []
#     r2F=[]
#     d_r2F=[]
#     for beta in betas:
#         path = f"/home/negro/projects/matching/step_scaling/L{Ns}/T42_L{Ns}_b{beta}"
            
#         tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
#         r.append(tmp1)
#         r2F.append(tmp2)
#         d_r2F.append(tmp3)
    
#     r = np.column_stack(r)
#     r2F = np.column_stack(r2F)
#     d_r2F = np.column_stack(d_r2F)
        
#     plt.errorbar(betas, r2F[0, :], d_r2F[0, :], **plot.data(1))
        
#     ## fitting 
#     # linear model
#     def model(x, a, b,c ):
#             return a + b*x +c*x**2
        
#     # data to fit
#     x_min = np.minimum(betas[0], np.min(x_ac))
#     x_max = betas[-1]
#     y = r2F[0, :]
#     d_y = d_r2F[0, :]
        
#     # starting parameters and bounds
#     p0 = [1,1,1]
#     bounds =  -np.inf, np.inf

#     # fitting
#     opt, cov = curve_fit(model, betas, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
#     # x_fit & y_fit
#     x_fit = np.linspace(x_min, x_max, 100)
#     y_fit = model(x_fit, opt[0], opt[1], opt[2])
    
#     # plotting the fit curve
#     plt.plot(x_fit, y_fit, **plot.fit(1), label = 'linear fit')
        
#     ## confidence band
#     # bootstrap samples
#     n_boot = 200

#     boot_opt = np.random.multivariate_normal(opt, cov, size=n_boot, tol=1e-10)
        
#     boot_y_fit = [model(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]        
#     boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
#     plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(1))
            
#     plt.grid (True, linestyle = '--', linewidth = 0.25)

#     plt.xlabel(r"$\beta$")
#     plt.ylabel(r"$r^2F(r,g)$")
    
#     plt.errorbar(x_ac[0], r2F_r1_ac[0], **plot.data(2), label = "l=1")
#     plt.errorbar(x_ac[1], r2F_r1_ac[1], **plot.data(3), label = "l=2")
#     plt.errorbar(x_ac[2], r2F_r1_ac[2], **plot.data(4), label = "l=3")
    
#     plt.legend()
    
#     plt.savefig("/home/negro/projects/matching/step_scaling/confronto_r2F_r1_L4.png", dpi=300, bbox_inches='tight')
    
#     plt.show()
    
# def confronto_r2F_r2_L4(Ns):
#     # [l=1, l=2, l=3]
#     x_ac = [1/0.36915425472851615**2, 1/0.7531542547285165**2, 1/0.7400542547285165**2]
#     r2F_r2_ac = [1.0226308779857438, 1.0248333693799925, 1.0655839649705927]
    
#     betas = [1.6, 1.8, 2, 4, 4.05, 4.15, 4.25]

#     plt.figure(figsize=(16,12))

#     r = []
#     r2F=[]
#     d_r2F=[]
#     for beta in betas:
#         path = f"/home/negro/projects/matching/step_scaling/L{Ns}/T42_L{Ns}_b{beta}"
            
#         tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
#         r.append(tmp1)
#         r2F.append(tmp2)
#         d_r2F.append(tmp3)
    
#     r = np.column_stack(r)
#     r2F = np.column_stack(r2F)
#     d_r2F = np.column_stack(d_r2F)
        
#     plt.errorbar(betas, r2F[1, :], d_r2F[1, :], **plot.data(1), label='lagrangian')
        
#     ## fitting 
#     # linear model
#     def model(x, a, b, c):
#             return a + b*x +c*x**2
        
#     # data to fit
#     x_min = np.minimum(betas[0], np.min(x_ac))
#     x_max = betas[-1]
#     y = r2F[1, :]
#     d_y = d_r2F[1, :]
        
#     # starting parameters and bounds
#     p0 = [1,1,1]
#     bounds =  -np.inf, np.inf

#     # fitting
#     opt, cov = curve_fit(model, betas, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
#     # x_fit & y_fit
#     x_fit = np.linspace(x_min, x_max, 100)
#     y_fit = model(x_fit, opt[0], opt[1], opt[2])
    
#     # plotting the fit curve
#     plt.plot(x_fit, y_fit, **plot.fit(1), label = 'quadratic fit')
        
#     ## confidence band
#     # bootstrap samples
#     n_boot = 200

#     boot_opt = np.random.multivariate_normal(opt, cov, size=n_boot, tol=1e-10)
        
#     boot_y_fit = [model(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]        
#     boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
#     plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(1))
            
#     plt.grid (True, linestyle = '--', linewidth = 0.25)

#     plt.xlabel(r"$\beta$")
#     plt.ylabel(r"$r^2F(r,g)$")
    
#     plt.errorbar(x_ac[0], r2F_r2_ac[0], **plot.data(2), label = "l=1")
#     plt.errorbar(x_ac[1], r2F_r2_ac[1], **plot.data(3), label = "l=2")
#     plt.errorbar(x_ac[2], r2F_r2_ac[2], **plot.data(4), label = "l=3")
    
#     plt.legend()
    
#     plt.savefig("/home/negro/projects/matching/step_scaling/confronto_r2F_r2_L4.png", dpi=300, bbox_inches='tight')
#     plt.show()
    
# def confronto_r2F_L3():
#     path = "/home/negro/projects/matching/step_scaling/L3/"

#     x = [1.0, np.sqrt(5)]
    
#     betas = [1.4, 1.]
    
#     # [l=1, l=2, l=3, l=4]
#     r2F_r1_ac = [0.29536698, 0.19739002867530875, 0.18957441018081542, 0.18940793972490294]
#     r2F_r2_ac = [1.02175514, 1.0250710849945603, 1.0648107364285249, 1.065914289433689]
    
#     plt.figure(figsize=(18,12))
    
#     for i, y in enumerate(r2F_r1_ac):
#         plt.errorbar(x[0], y, **plot.data(i+1, label = f"l={i+1}"))
    
#     for i, y in enumerate(r2F_r2_ac):
#         plt.errorbar(x[1], y, **plot.data(i+1))
        
#     for i, beta in enumerate([1.8, 1.9, 2]):
#         r2F, d_r2F = np.loadtxt(f"{path}/T42_L3_b{beta}/analysis/r2F.txt", usecols=(1,2), unpack=True)
#         plt.errorbar(x, r2F, d_r2F, **plot.data(0), label = ("lagrangian" if i==0 else ""))
        
#     plt.grid (True, linestyle = '--', linewidth = 0.25)

#     plt.xlabel(r'$r$')
#     plt.ylabel(fr'$r^2F(r,1/g^2)$')
    
#     plt.legend()
#     plt.savefig("/home/negro/projects/matching/step_scaling/matching_beta_Ns3.png", dpi=300, bbox_inches='tight')

#     plt.show()

# def plot_AC():
    
#     beta_ac_Ns3 = [1.4]
    
#     r2F_r1_ac_Ns3 = [0.29536698, 0.19739002867530875, 0.18957441018081542, 0.18940793972490294]
#     r2F_r2_ac_Ns3 = [1.02175514, 1.0250710849945603, 1.0648107364285249, 1.065914289433689]
    
#     beta_ac_Ns4 = [1/0.36915425472851615**2, 1/0.7531542547285165**2, 1/0.7400542547285165**2]

#     r2F_r1_ac_Ns4 = [0.14874300045572017, 0.2400338548299101, 0.20420630542747306]
#     r2F_r2_ac_Ns4 = [1.0226308779857438, 1.0248333693799925, 1.0655839649705927]
    
#     plt.figure(figsize=(18,12))
    
#     for i in [2]:
#         plt.errorbar(beta_ac_Ns3, r2F_r1_ac_Ns3[i], **plot.data_nomarker(1), marker = "v" , label=fr"$r_1=1$, $N_s=3$")
#         plt.errorbar(beta_ac_Ns3, r2F_r2_ac_Ns3[i], **plot.data_nomarker(1), marker = "^", label=fr"$r_2=\sqrt{5}$, $N_s=3$")
#         plt.errorbar(beta_ac_Ns4[i], r2F_r1_ac_Ns4[i], **plot.data_nomarker(2), marker = "D", label=fr"$r_1=\sqrt{2}$, $N_s=4$")
#         plt.errorbar(beta_ac_Ns4[i], r2F_r2_ac_Ns4[i], **plot.data_nomarker(2), marker = "o", label=fr"$r_2=\sqrt{10}$, $N_s=4$")

#     plt.title(f"l={i+1}")   
        
#     plt.xlabel(r"$l$")
#     plt.ylabel(fr'$r^2F(r,1/g^2)$')
    
#     plt.grid (True, linestyle = '--', linewidth = 0.25)
        
#     plt.legend()
#     plt.savefig(f"/home/negro/projects/matching/step_scaling/plots_AC/plot.png", dpi=300, bbox_inches='tight')

#     plt.show()

# def tuning_barecoup():
#     path = "/home/negro/projects/matching/step_scaling/L3/"
    
#     betas = [1.4, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975, 2]
    
#     # [l=1, l=2, l=3, l=4]
#     r2F_r1_ac = [0.29536698, 0.19739002867530875, 0.18957441018081542, 0.18940793972490294]
#     r2F_r2_ac = [1.02175514, 1.0250710849945603, 1.0648107364285249, 1.065914289433689]
    
#     plt.figure(figsize=(18,12))

#     plt.errorbar(betas[0], r2F_r1_ac[3], **plot.data(0), label = "hamiltonian, l=4")
    
#     r2F_r1 = []
#     d_r2F_r1 = []
#     for i, beta in enumerate(betas[1:]):
#         foo, bar = np.loadtxt(f"{path}/T48_L3_b{beta}/analysis/r2F.txt", usecols=(1,2), unpack=True)
#         r2F_r1.append(foo[0])
#         d_r2F_r1.append(bar[0])
    
#     plt.errorbar(betas[1:], r2F_r1, d_r2F_r1, **plot.data(1), label = ("lagrangian" if i==0 else ""))
    
#     def quadratic(x, a, b, c):
#         return a + b*x + c*x**2
    
#     opt, cov, x_fit, y_fit, boot_band, chi2 = reg.fit_with_scipy(betas[1:], r2F_r1, d_r2F_r1, quadratic, [1,1,1], mask=None)
    
#     a, b, c = opt[0], opt[1], opt[2]
#     y0 = r2F_r1_ac[3]

#     discriminant = b**2 - 4 * c * (a - y0)
#     if discriminant < 0:
#         raise ValueError("No real solution — the curve never reaches this value.")

#     x1 = (-b + np.sqrt(discriminant)) / (2 * c)
#     x2 = (-b - np.sqrt(discriminant)) / (2 * c)
    
#     plt.plot(x_fit, y_fit, **plot.fit(1), label="lagrangian")
#     plt.fill_between(x_fit, y_fit-boot_band, y_fit+boot_band, **plot.conf_band(1))
    
#     plt.plot([1.4, x2],[r2F_r1_ac[3], r2F_r1_ac[3]], **plot.fit(5))
    
#     rng = np.random.default_rng(seed=8220)
#     boot_opt_sp = rng.multivariate_normal(opt, cov, size=1000, tol=1e-10)
    
#     boot_y_fit_sp = np.array([quadratic(x2, *params) for params in boot_opt_sp])
#     err = np.std(boot_y_fit_sp, axis=0, ddof=1)
        
#     plt.errorbar(x2, quadratic(x2, *opt), err, **plot.data(5))
    
#     plt.grid (True, linestyle = '--', linewidth = 0.25)
#     plt.xlabel(f"$1/g^2$")
#     plt.ylabel(f"$r^2F(r_1,1/g^2)$")
        
#     plt.legend()
#     plt.savefig("/home/negro/projects/matching/step_scaling/tuning_barecoup_Ns3_r1.png", dpi=300, bbox_inches='tight')
    
#     return x2

# def deduce_runcoup_r2(x2):
#     def fit_r2(x, a, b):
#         return a + b*x
    
#     path = "/home/negro/projects/matching/step_scaling/L3/"

#     betas = [1.4, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975, 2]
#     r2F_r2_ac = 1.065914289433689
    
#     r2F_r2 = []
#     d_r2F_r2 = []
#     for beta in betas[1:]:
#         foo, bar = np.loadtxt(f"{path}/T48_L3_b{beta}/analysis/r2F.txt", usecols=(1,2), unpack=True)
#         r2F_r2.append(foo[1])
#         d_r2F_r2.append(bar[1])
        
#     plt.figure(figsize = (18,12))

#     plt.errorbar(betas[0], r2F_r2_ac, **plot.data(0), label="hamiltonian, l=4")
    
#     plt.errorbar(betas[1:], r2F_r2, d_r2F_r2, **plot.data(1), label = "lagrangian" )
    
#     opt, cov, x_fit, y_fit, boot_band, chi2red = reg.fit_with_scipy(betas[1:], r2F_r2, d_r2F_r2, fit_r2, [1,1], mask=None)
    
#     plt.plot(x_fit, y_fit, **plot.fit(1), label = fr"$\chi^2_r$={chi2red:.2f}")
#     plt.fill_between(x_fit, y_fit-boot_band, y_fit+boot_band, **plot.conf_band(1))
    
#     rng = np.random.default_rng(seed=8220)
#     boot_opt_sp = rng.multivariate_normal(opt, cov, size=1000, tol=1e-10)
    
#     boot_y_fit_sp = np.array([fit_r2(x2, *params) for params in boot_opt_sp])
#     err = np.std(boot_y_fit_sp, axis=0, ddof=1)
        
#     plt.errorbar(x2, fit_r2(x2, *opt), err, **plot.data(5))
    
#     plt.plot([1.4, x2],[fit_r2(x2, *opt), fit_r2(x2, *opt)], **plot.fit(5))
#     plt.fill_between(np.linspace(1.4, x2, 100),fit_r2(x2, *opt)-err, fit_r2(x2, *opt)+err, **plot.conf_band(5))
    
#     plt.grid (True, linestyle = '--', linewidth = 0.25)
#     plt.xlabel(f"$1/g^2$")
#     plt.ylabel(f"$r^2F(r_2,1/g^2)$")
        
#     plt.legend()
    
#     plt.savefig("/home/negro/projects/matching/step_scaling/deduce_barecoup_Ns3_r2.png", dpi=300, bbox_inches='tight')
 
#  ## main
 
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
    ws_max = 21
    
    if not os.path.isdir(f"{path}/analysis/therm_{type}/"):
        thermalization(path, wt_max, ws_max, type = type)
        
    if not os.path.isdir(f"{path}/analysis/bsa_{type}/"):
        bsa_potential(path, wt_max, ws_max, type = type)
    
    ws_max_plot = 21
    wt_max_plot = [36] * ws_max
    
    if not os.path.isfile(f"{path}/analysis/staircase_potential_wt.npy"):
        get_potential(path, wt_max, ws_max, wsplot=wsplot, type=type)
    
    plot_potential(path, wt_max_plot, ws_max_plot, type=type)

    wt_min_fit = [8] * ws_max_plot
    wt_max_fit = [36] * ws_max_plot
    
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
    Ns = 5
    beta = 12
    
    path_glob = f"/home/negro/projects/matching/step_scaling/CLSS/Nt48_Ns{Ns}/b{beta}"
    
    if not os.path.isfile(f"{path_glob}/analysis/Wloop.dat"):
            if not os.path.isfile(f"{path_glob}/analysis/sWloop.dat"):
                concatenate.concatenate(f"{path_glob}/data", 2000, f"{path_glob}/analysis")
    
    ## planar
    #planarpot(path_glob)

    ## staircase
    wsplot = np.array([np.sqrt(i**2 + j**2) for i in range(1,Ns) for j in range(1,i+1)])

    #staircase(path_glob, wsplot=wsplot)

    #plotfit_potential(path_glob)
    
    compute_r2F(path_glob, Ns=Ns)
    
    x = [3, 4, 12, 12]
    y = [0.424, 0.489, 0.52, 0.404]
    d_y = [0.062, 0.088, 0.050, 0.048]
    plt.figure()
    plt.errorbar(x, y, d_y, **plot.data(4))
    plt.show()