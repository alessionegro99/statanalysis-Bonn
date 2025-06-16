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

def readfile(path,extra=''):
    output = np.loadtxt(f'{path}/analysis/dati{extra}.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    # plaqs plaqt ReP ImP W(wt=1,ws=1) W(Wt=1,ws=2) ... W(wt=1,ws=10) W(wt=2,ws=10) ... W(wt=10,ws=10)
    
    return np.column_stack(columns)

def thermalization(path, wsmax, ws, wt): 
    data = readfile(path,'_therm')
    
    wloop = data[:, 4 + ws + wsmax*wt]
    
    x = np.arange(0,len(wloop), len(wloop)//500)
    y = wloop[0:len(wloop):len(wloop)//500]

    plt.figure(figsize=(16,12))
    plt.plot(x, y
             , marker = 'o'
             , linestyle = '-', linewidth = 0.375
             , markersize = 2
             , color = plot.color_dict[1])
    
    plt.xlabel(r'$t_i$')
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'MC history')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.savefig(f"{path}/analysis/thermalization.png", dpi=300, bbox_inches='tight')
    
def blocksize_analysis_primary(path, wsmax, ws, wt):
    data = readfile(path)
    
    wloop = data[:, 4 + ws + wsmax*wt]

    boot.blocksize_analysis_primary(wloop, 200, [10, 500, 5], savefig=1, path=f"{path}/analysis/")
    
def plot_potential_Wilsont(path):
    data = readfile(path)
    
    def id(x):
        return x
    
    seed = 8220
    samples = 200
    blocksize = 50
    
    wsmax = 10
    
    wtmaxplot = 10
    wsplot = range(1, wsmax+1)
    
    np.savetxt(f"{path}/analysis/potential_wt.txt", np.arange(1,wsmax+1).reshape(-1,1))
    
    savefoo = np.loadtxt(f"{path}/analysis/potential_wt.txt").reshape(-1,1)
    
    plt.figure(figsize=(16,12))
    for wt in range(wtmaxplot):
        pb.progress_bar(wt, wtmaxplot)
        V = []
        d_V = []
        
        for ws in range(wsmax):
            wloop = data[:, 4 + ws + wsmax*wt] 
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)

            args = [id, wloop]
            
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, samples, 0, args, seed=seed)
            V.append(ris) # rifare
            d_V.append(err) 
                
        savefoo = np.hstack((savefoo, np.array(V).reshape(-1,1), np.array(d_V).reshape(-1,1)))
        
        plt.errorbar(wsplot, V, d_V, **plot.data(wt), label=fr'$w_t={wt+1}$')
        
    np.savetxt(f"{path}/analysis/potential_wt.txt", savefoo)
        
    plt.xlabel(r'$w_s$')
    plt.ylabel(r'$aV(w_s)$')
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/potential_wt.png", dpi=300, bbox_inches='tight')

def plot_potential_ws(path):
    
    wtmax = 10
    
    data = np.loadtxt(f"{path}/analysis/potential_wt.txt")
    
    ws = data[:,0]
    
    pot = []
    d_pot = []
    for i in range(wtmax):
        pot.append(data[:, 1 + i*2])
        d_pot.append(data[:, 2 + i*2])
        
    pot = np.array(list(map(list, zip(*np.array(pot)))))
    d_pot = np.array(list(map(list, zip(*np.array(d_pot)))))
    
    plt.figure(figsize=(16,12))
    
    plt.xlabel(r'$w_s$')
    plt.ylabel(r'$aV(w_s)$')
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$')
    
    for i, (pota, d_pota) in enumerate(zip(pot, d_pot)):
        plt.errorbar(1/ws, pota, d_pota, **plot.data(i), label=fr"$w_t={i+1}$")
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/potential_ws.png", dpi=300, bbox_inches='tight')
    
def fit_potential_ws(path):
    min_list = [5, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    max_list = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    
    def model(x, a, b):
        return a + b/x
    
    def linearized_model(z, A, B):
        return A + B*z
    
    wtmax = 10
    
    data = np.loadtxt(f"{path}/analysis/potential_wt.txt")
    
    ws = data[:,0]
    
    pot = []
    d_pot = []
    for i in range(wtmax):
        pot.append(data[:, 1 + i*2])
        d_pot.append(data[:, 2 + i*2])
        
    pot = np.array(list(map(list, zip(*np.array(pot)))))
    d_pot = np.array(list(map(list, zip(*np.array(d_pot)))))
    
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_s$')
    plt.ylabel(r'$aV(w_s)$')
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$')
    
    bounds = -np.inf, np.inf
    
    p0 = [1, 1]
    
    ws_lst = []
    V0_lst = []
    d_V0_lst = []
    chi2red_lst = [] 
    
    for i, (pota, d_pota) in enumerate(zip(pot, d_pot)):
        plt.errorbar(1/ws, pota, d_pota, **plot.data(i), label=fr"$w_t={i+1}$")
        
        min = min_list[i]
        max = max_list[i]
        
        x = ws[min:max]
        y = pota[min:max]
        d_y = d_pota[min:max]
        
        opt, cov = curve_fit(model, x, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
        chi2 = reg.chi2_corr(x, y, model, np.diag(d_y**2), opt[0], opt[1])
        chi2red = chi2/(len(x) - len(opt))
        
        z_fit = np.linspace(0, 1/x[0], 200)
        y_fit = linearized_model(z_fit, opt[0], opt[1])
        
        boot_opt = np.random.multivariate_normal(opt, cov, size=200, tol=1e-10)
        
        boot_y_fit = [linearized_model(z_fit, boot_opt[i,0], boot_opt[i,1]) for i in range(200)]        
        boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
            
        plt.fill_between(z_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(i))
        plt.plot(z_fit, y_fit, **plot.fit(i))
        
        ws_lst.append(i+1)
        V0_lst.append(opt[0])
        d_V0_lst.append(cov[1][1]**0.5)
        chi2red_lst.append(chi2red)
        
    savefoo = [ws_lst, V0_lst, d_V0_lst, chi2red_lst]
    savefoo = np.column_stack(savefoo)
    
    np.savetxt(f"{path}/analysis/extrap_potential_ws.txt", savefoo)

    plt.ylim(0,1.4)
    plt.xlim(-0.01,0.4)
    
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/extrap_potential_ws.png", dpi=300, bbox_inches='tight')
    plt.show()

def fit_V(path):
    
    Cmin = 0
    Cmax = 10
    
    Lmin = 0
    Lmax = 10
    
    def Cornell(x, a, b, c):
        return a + b/x + c*x
    
    def logpot(x, a, b, c):
        return a + np.log(x)*b + c*x
    
    data = np.loadtxt(f"{path}/analysis/extrap_potential_ws.txt")
    
    x = data[:,0]
    y = data[:,1]
    d_y = data[:,2]
    
    plt.figure(figsize=(18,12))
    plt.errorbar(x, y, d_y, **plot.data(0))
    
    p0 = [1,1,1]
    bounds  = -np.inf, np.inf
    
    ##Cornell
    x_Cornell, y_Cornell, d_y_Cornell = x[Cmin:Cmax], y[Cmin:Cmax], d_y[Cmin:Cmax]

    opt, cov = curve_fit(Cornell, x_Cornell, y_Cornell, sigma=d_y_Cornell, absolute_sigma=True, p0=p0, bounds=bounds)
    
    x_fit = np.linspace(x_Cornell[0], x_Cornell[-1], 100)
    y_fit = Cornell(x_fit, opt[0], opt[1], opt[2])
    
    plt.plot(x_fit, y_fit, **plot.fit(1), label="Cornell")
    
    boot_opt = np.random.multivariate_normal(opt, cov, size=200, tol=1e-10)
        
    boot_y_fit = [Cornell(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(200)]        
    boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
            
    plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(1))
    
    chi2 = reg.chi2_corr(x_Cornell, y_Cornell, Cornell, np.diag(d_y_Cornell**2), opt[0], opt[1], opt[2])
    chi2red = chi2/(len(x_Cornell) - len(opt))
    
    print(opt, chi2red)
        
    ## logpot
    x_logpot, y_logpot, d_y_logpot = x[Lmin:Cmax], y[Lmin:Cmax], d_y[Lmin:Lmax]

    opt, cov = curve_fit(logpot, x_logpot, y_logpot, sigma=d_y_logpot, absolute_sigma=True, p0=p0, bounds=bounds)

    x_fit = np.linspace(x_logpot[0], x_logpot[-1], 100)
    y_fit = logpot(x_fit, opt[0], opt[1], opt[2])
    
    plt.plot(x_fit, y_fit, **plot.fit(2), label = "log")
    
    boot_opt = np.random.multivariate_normal(opt, cov, size=200, tol=1e-10)
        
    boot_y_fit = [logpot(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(200)]        
    boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
            
    plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(2))
    
    chi2 = reg.chi2_corr(x_logpot, y_logpot, logpot, np.diag(d_y_logpot**2), opt[0], opt[1], opt[2])
    chi2red = chi2/(len(x_logpot) - len(opt))
    
    print(opt, chi2red)
    
    plt.xlabel(r"$w_s$")
    plt.ylabel(r"$aV(w_s)$")    
    
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
        
    # plt.savefig(f"{path}/analysis/Cornell_vs_log.png", dpi=300, bbox_inches='tight')

    # plt.show()

    def force(x, b, s):
        return  b/x + s
    
    x_fit = np.linspace(1,10,100)
    y_fit = force(x_fit, opt[1], opt[2])
    
    plt.plot(x_fit, y_fit, **plot.fit(4), label = "force")
            
    plt.savefig(f"{path}/analysis/Cornell_vs_log_vs_force.png", dpi=300, bbox_inches='tight')
       
    boot_opt = np.random.multivariate_normal(opt[1:], cov[1:,1:], size=200, tol=1e-10)
        
    boot_y_fit = [force(x_fit, boot_opt[i,0], boot_opt[i,1]) for i in range(200)]        
    boot_band = np.std(boot_y_fit, axis = 0, ddof=1)
    
    plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(4))

    plt.legend()

    #plt.show()
    
    Fr1 = force(1, opt[1], opt[2])
    Fr2 = force(np.sqrt(2), opt[1], opt[2])
    
    
    
if __name__ == "__main__":
    path = "/home/negro/projects/matching/step_scaling/infvol_PBC/T42_L42_b2"
    
    #thermalization(path, 10, 5, 5)
    
    #concatenate.concatenate(f"{path}/data", 500)
    
    #blocksize_analysis_primary(path, 10, 5, 5)
    
    #plot_potential_Wilsont(path)
    
    #plot_potential_ws(path)

    #fit_potential_ws(path)
    
    fit_V(path)