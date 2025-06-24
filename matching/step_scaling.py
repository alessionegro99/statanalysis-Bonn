import sys
import os
import shutil

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

    # plaqs plaqt W(wt=1,ws=1) ... W(wt=10,ws=1) W(wt=1,ws=sqrt(5)) ... W(wt=10,ws=sqrt(5)) W(wt=10,ws=1) W(wt=1,ws=sqrt(8)) ... W(wt=10,ws=sqrt(8))
    
    return np.column_stack(columns)
    
def thermalization(path): 
    output = np.loadtxt(f'{path}/data/dati_0.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    # plaqs plaqt W(wt=1,ws=1) ... W(wt=10,ws=1) W(wt=1,ws=sqrt(5)) ... W(wt=10,ws=sqrt(5)) W(wt=10,ws=1) W(wt=1,ws=sqrt(8)) ... W(wt=10,ws=sqrt(8))
    
    data = np.column_stack(columns)
    
    foo = data[:,4 + 5 + 3*5]
    
    x = np.arange(0,len(foo), len(foo)//500)
    y = foo[0:len(foo):len(foo)//500]

    plt.figure(figsize=(18,12))
    plt.plot(x, y
             , marker = 'o'
             , linestyle = '-', linewidth = 0.375
             , markersize = 2
             , color = plot.color_dict[1])
    
    plt.xlabel(r'$t_i$')
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.savefig(f"{path}/analysis/thermalization.png", dpi=300, bbox_inches='tight')
        
def blocksize_analysis_primary(path):
    data = readfile(path)
    
    obs=data[:,5 + 3*5]
    boot.blocksize_analysis_primary(obs, 100, [10, 500, 5], savefig=1, path=f"{path}/analysis/")
 
def blocksize_analysis_secondary(path):
    data = readfile(path)
    
    wt = 4
    
    def potential(x):
        eps=1e-10
        return -np.log(np.clip(x, eps, None))/wt # remember 1/wt
    
    def id(x):
        return x
    
    W = data[:,1 + wt + 10]
    
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

def get_potential_wt(path, wsplot, wtmax):
    data = readfile(path)
    def id(x):
        return x
    
    seed = 8220
    samples = 200
    blocksizes = [50, 50, 50]
    
    wsmax = 3
    
    wtmaxplot = wtmax
    
    wsplot = np.array(wsplot)
    
    V = []
    d_V = []
    V_bs = []
    
    plt.figure(figsize=(18,12))
    for wt in range(wtmaxplot):
        V_wt = []
        d_V_wt = []
        V_bs_wt = []
        
        for ws, blocksize in zip(range(wsmax), blocksizes):
            W = data[:, 2 + wt + wtmax*ws]

            args = [id, W]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            print(f"Currently bootstrapping w_t={wt+1}, w_s={wsplot[ws]:.2f}...")
            _, err, bs = boot.bootstrap_for_secondary(potential, blocksize, samples, 1, args, seed=seed, returnsamples=1)
        
            sys.stdout.flush()

            V_wt.append(potential(np.mean(W)))
            d_V_wt.append(err)
            V_bs_wt.append(bs)
        
        plt.errorbar(wsplot, V_wt, d_V_wt, **plot.data(wt), label=fr'$w_t={wt+1}$')
        
        V.append(V_wt)
        d_V.append(d_V_wt)
        V_bs.append(V_bs_wt)
        
    np.save(f"{path}/analysis/potential_wt", np.array([wsplot, V, d_V, V_bs], dtype=object))
            
    plt.xlabel(r'$w_s$')
    plt.ylabel(r'$aV(w_s)$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/potential_wt.png", dpi=300, bbox_inches='tight')        
        
def plot_effmass(path, wtmax):
    data = np.load(f"{path}/analysis/potential_wt.npy", allow_pickle=True)
    
    wsplot, effm, d_effm, effm_bs = map(np.array, data[:4])
    
    #print(d_effm[:,0])
    #print(np.std(effm_bs[:,0,:], axis = 1))
        
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$')

    for i, (effma, d_effma) in enumerate(zip(np.transpose(effm), np.transpose(d_effm))):
        plt.errorbar(np.arange(1,wtmax+1), effma, d_effma, **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/effmass.png", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$')

    for i, (effma, d_effma) in enumerate(zip(np.transpose(effm), np.transpose(d_effm))):
        plt.errorbar(1/np.arange(1,wtmax+1), effma, d_effma, **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/potential_ws.png", dpi=300, bbox_inches='tight')

def fit_potential_ws(path, wsplot, minlist, maxlist):
    
    def model(x, a, b):
        return a + b/x
    
    def linearized_model(z, A, B):
        return A + B*z
    
    wtmax = 10
    
    wt = np.arange(1, wtmax + 1)
    
    data = np.loadtxt(f"{path}/analysis/potential_wt.txt")
        
    pot = []
    d_pot = []
    for i in range(wtmax):
        pot.append(data[:, 1 + i*2])
        d_pot.append(data[:, 2 + i*2])
        
    pot = np.array(list(map(list, zip(*np.array(pot)))))
    d_pot = np.array(list(map(list, zip(*np.array(d_pot)))))
    
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$')
    
    bounds = -np.inf, np.inf
    
    p0 = [1, 1]
    
    boot_V0 = []
    V0_lst = []
    d_V0_lst = []
    chi2red_lst = [] 
    
    for i, (pota, d_pota) in enumerate(zip(pot, d_pot)):
        plt.errorbar(1/np.arange(1,wtmax+1), pota, d_pota, **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
        
        min = min_list[i]
        max = max_list[i]
        
        x = wt[min:max]
        y = pota[min:max]
        d_y = d_pota[min:max]
        
        opt, cov = curve_fit(model, x, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
        chi2 = reg.chi2_corr(x, y, model, np.diag(d_y**2), opt[0], opt[1])
        chi2red = chi2/(len(x) - len(opt))
        
        z_fit = np.linspace(0, 1/x[0], 200)
        y_fit = linearized_model(z_fit, opt[0], opt[1])
        
        n_boot = 5000
        
        rng = np.random.default_rng(seed=8220)  
        boot_opt = rng.multivariate_normal(opt, cov, size=n_boot, tol=1e-10)
        
        boot_V0.append(boot_opt[:, 0])   
        
        boot_y_fit = [linearized_model(z_fit, boot_opt[i,0], boot_opt[i,1]) for i in range(n_boot)]
        boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
            
        plt.fill_between(z_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(i))
        plt.plot(z_fit, y_fit, **plot.fit(i))
        
        V0_lst.append(opt[0])
        d_V0_lst.append(cov[0][0]**0.5)
        chi2red_lst.append(chi2red)
        
        print(f"V={opt[0]}, d_V={cov[0][0]**0.5}, chi2red={chi2red}")
        
    savefoo = [np.array(wsplot), V0_lst, d_V0_lst, chi2red_lst]
    savefoo = np.column_stack(savefoo)

    np.savetxt(f"{path}/analysis/extrap_potential_ws.txt", savefoo)
    np.savetxt(f"{path}/analysis/boot_potential_ws.txt", boot_V0)

    plt.xlim(-0.01,0.4)
    
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/extrap_potential_ws.png", dpi=300, bbox_inches='tight')

def compute_r2F(path, wsplot):    
    pot = np.loadtxt(f"{path}/analysis/extrap_potential_ws.txt", usecols=(1), unpack=True)
    boot_pot = np.loadtxt(f"{path}/analysis/boot_potential_ws.txt")

    r2F_r1 = wsplot[0]**2*(pot[1]-pot[0])/(wsplot[1]-wsplot[0])
    r2F_r2 = wsplot[1]**2*(pot[2]-pot[1])/(wsplot[2]-wsplot[1])
    
    boot_r2F_r1 = wsplot[0]**2*(boot_pot[:,1]-boot_pot[:,0])/(wsplot[1]-wsplot[0])
    boot_r2F_r2 = wsplot[1]**2*(boot_pot[:,2]-boot_pot[:,1])/(wsplot[2]-wsplot[1])
    
    d_r2F_r1 = np.std(boot_r2F_r1, ddof=1)
    d_r2F_r2 = np.std(boot_r2F_r2, ddof=1)
    
    data = np.column_stack((np.array([wsplot[0], wsplot[1]]), np.array([r2F_r1, r2F_r2]), np.array([d_r2F_r1, d_r2F_r2])))
    
    np.savetxt(f"{path}/analysis/r2F.txt", data)
    
def tune_r2F():
    Ns_list=[4, 5]
    
    stuff = []
    runcoup_r1 = []
    d_runcoup_r1 = []
    
    betas_list = [[1.6, 1.8, 2, 4, 4.05, 4.15, 4.25], [11, 11.5, 12, 12.5]]

    plt.figure(figsize=(16,12))

    r = []
    r2F=[]
    d_r2F=[]
    
    path = f"/home/negro/projects/matching/step_scaling/L3/T42_L3_b3"
    
    ## r = r2
    tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
    r.append(tmp1)
    r2F.append(tmp2)
    d_r2F.append(tmp3)
    
    r = np.column_stack(r)
    r2F = np.column_stack(r2F)
    d_r2F = np.column_stack(d_r2F)
    
    r2F_L3 = r2F[:,:]
    
    plt.errorbar(3, r2F[1, :], d_r2F[1,:], **plot.data(0))
    
    for count, Ns in enumerate(Ns_list):
        r = []
        r2F=[]
        d_r2F=[]
                
        betas = betas_list[count]
        for beta in betas:
            path = f"/home/negro/projects/matching/step_scaling/L{Ns}/T42_L{Ns}_b{beta}"
            
            tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
            r.append(tmp1)
            r2F.append(tmp2)
            d_r2F.append(tmp3)
    
        r = np.column_stack(r)
        r2F = np.column_stack(r2F)
        d_r2F = np.column_stack(d_r2F)
        
        plt.errorbar(betas, r2F[1, :], d_r2F[1, :], **plot.data(count+1))
        
        ## fitting 
        # quadratic model
        def model(x, a, b, c):
            return a + b*x +c*x**2
        
        # data to fit
        x = betas
        y = r2F[1, :]
        d_y = d_r2F[1, :]
        
        # starting parameters and bounds
        p0 = [1,1,1]
        bounds =  -np.inf, np.inf

        # fitting
        opt, cov = curve_fit(model, betas, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
        # x_fit & y_fit
        x_fit = np.linspace(x[0], x[-1], 100)
        y_fit = model(x_fit, opt[0], opt[1], opt[2])

        # plotting the fit curve
        plt.plot(x_fit, y_fit, **plot.fit(count+1), label = 'null')
        
        ## confidence band
        # bootstrap samples
        n_boot = 200

        boot_opt = np.random.multivariate_normal(opt, cov, size=n_boot, tol=1e-10)
        
        boot_y_fit = [model(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]        
        boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
        plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(count + 1))
        
        ## getting tuned point and error
        def func(x):
            return (model(x, opt[0], opt[1], opt[2]) - r2F_L3[1,:])
        beta0 = 4
        beta_tuned = fsolve(func, beta0)
        
        stuff.append(beta_tuned)
        
        r2F_tuned = model(beta_tuned, opt[0], opt[1], opt[2])
        boot_r2F_tuned = [model(beta_tuned, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]
        d_r2F_tuned = np.std(boot_r2F_tuned, axis = 0, ddof=1)
        
        plt.errorbar(beta_tuned, r2F_tuned, d_r2F_tuned, **plot.data(0))
        
    ## r = r1
    r = []
    r2F=[]
    d_r2F=[]
    
    tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
    r.append(tmp1)
    r2F.append(tmp2)
    d_r2F.append(tmp3)
    
    r = np.column_stack(r)
    r2F = np.column_stack(r2F)
    d_r2F = np.column_stack(d_r2F)
    
    plt.errorbar(3, r2F[0, :], d_r2F[0,:], **plot.data(0))
    
    r2F_L3 = r2F[:,:]

    for count, Ns in enumerate(Ns_list):
        r = []
        r2F=[]
        d_r2F=[]
                
        betas = betas_list[count]
        for beta in betas:
            path = f"/home/negro/projects/matching/step_scaling/L{Ns}/T42_L{Ns}_b{beta}"
            
            tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
            r.append(tmp1)
            r2F.append(tmp2)
            d_r2F.append(tmp3)
    
        r = np.column_stack(r)
        r2F = np.column_stack(r2F)
        d_r2F = np.column_stack(d_r2F)
        
        plt.errorbar(betas, r2F[0, :], d_r2F[0, :], **plot.data(count+1))
        
        ## fitting 
        # quadratic model
        def model(x, a, b, c):
            return a + b*x +c*x**2
        
        # data to fit
        x = betas
        y = r2F[0, :]
        d_y = d_r2F[0, :]
        
        # starting parameters and bounds
        p0 = [1,1,1]
        bounds =  -np.inf, np.inf

        # fitting
        opt, cov = curve_fit(model, betas, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
        # x_fit & y_fit
        x_fit = np.linspace(x[0], x[-1], 100)
        y_fit = model(x_fit, opt[0], opt[1], opt[2])

        # plotting the fit curve
        plt.plot(x_fit, y_fit, **plot.fit(count+1), label = 'null')
        
        ## confidence band
        # bootstrap samples
        n_boot = 200

        boot_opt = np.random.multivariate_normal(opt, cov, size=n_boot, tol=1e-10)
        
        boot_y_fit = [model(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]        
        boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
        plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(count + 1))
        
        ## getting tuned point and error
        def func(x):
            return (model(x, opt[0], opt[1], opt[2]) - r2F_L3[0,:])
        beta_tuned = stuff[count]
        
        r2F_tuned = model(beta_tuned, opt[0], opt[1], opt[2])
        boot_r2F_tuned = [model(beta_tuned, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]
        d_r2F_tuned = np.std(boot_r2F_tuned, axis = 0, ddof=1)
        
        runcoup_r1.append(r2F_tuned)
        d_runcoup_r1.append(d_r2F_tuned)
        
        plt.errorbar(beta_tuned, r2F_tuned, d_r2F_tuned, **plot.data(0))
        
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$r^2F(r,g)$", rotation=0)
    #plt.savefig("/home/negro/projects/matching/step_scaling/r2F_tune_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    tofile = np.column_stack((np.array(stuff), np.array(runcoup_r1), np.array(d_runcoup_r1)))

    np.savetxt("/home/negro/projects/matching/step_scaling/r2F_tune.txt", tofile)
    
def plot_r2F_vs_rlatt(path):
    
    r_latt = [1.00, 2**0.5, 5**0.5]
    
    r2F = []
    d_r2F = []
    
    # L=3
    L3_path = f"/home/negro/projects/matching/step_scaling/L3/T42_L3_b3"
    tmp1, tmp2 = np.loadtxt(f"{L3_path}/analysis/r2F.txt", usecols=(1,2), unpack=True)

    print(tmp1[0], tmp2[0])
    
    r2F.append(tmp1[0])
    d_r2F.append(tmp2[0])

    ## L\in{4,5}
    plt.figure(figsize = (16,12))
    
    plt.xlabel(r'$1/r^2_{latt}$')
    plt.ylabel(r'$r^2F(r_1,g)$', rotation = 0)

    tmp1, tmp2 = np.loadtxt(f"{path}/r2F_tune.txt", usecols=(1,2), unpack=True)
    
    r2F.append(tmp1[0])
    r2F.append(tmp1[1])

    d_r2F.append(tmp2[0])
    d_r2F.append(tmp2[1])
    
    r_latt = np.array(r_latt)
    r2F = np.array(r2F)
    d_r2F = np.array(d_r2F)
    
    plt.errorbar(1/r_latt**2, r2F, d_r2F, **plot.data(1))
    plt.gca().yaxis.set_label_coords(-0.1, 0.45)
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    
    plt.savefig(f'{path}/r2F_r1.png', dpi=300, bbox_inches='tight')
    plt.show()

def confronto_r2F_r1_L4(Ns):
    
    # [l=1, l=2, l=3]
    x_ac = [1/0.36915425472851615**2, 1/0.7531542547285165**2, 1/0.7400542547285165**2]
    r2F_r1_ac = [0.14874300045572017, 0.2400338548299101, 0.20420630542747306]
    
    betas = [1.6, 1.8, 2, 4, 4.05, 4.15, 4.25]

    plt.figure(figsize=(16,12))

    r = []
    r2F=[]
    d_r2F=[]
    for beta in betas:
        path = f"/home/negro/projects/matching/step_scaling/L{Ns}/T42_L{Ns}_b{beta}"
            
        tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
        r.append(tmp1)
        r2F.append(tmp2)
        d_r2F.append(tmp3)
    
    r = np.column_stack(r)
    r2F = np.column_stack(r2F)
    d_r2F = np.column_stack(d_r2F)
        
    plt.errorbar(betas, r2F[0, :], d_r2F[0, :], **plot.data(1))
        
    ## fitting 
    # linear model
    def model(x, a, b,c ):
            return a + b*x +c*x**2
        
    # data to fit
    x_min = np.minimum(betas[0], np.min(x_ac))
    x_max = betas[-1]
    y = r2F[0, :]
    d_y = d_r2F[0, :]
        
    # starting parameters and bounds
    p0 = [1,1,1]
    bounds =  -np.inf, np.inf

    # fitting
    opt, cov = curve_fit(model, betas, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
    # x_fit & y_fit
    x_fit = np.linspace(x_min, x_max, 100)
    y_fit = model(x_fit, opt[0], opt[1], opt[2])
    
    # plotting the fit curve
    plt.plot(x_fit, y_fit, **plot.fit(1), label = 'linear fit')
        
    ## confidence band
    # bootstrap samples
    n_boot = 200

    boot_opt = np.random.multivariate_normal(opt, cov, size=n_boot, tol=1e-10)
        
    boot_y_fit = [model(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]        
    boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
    plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(1))
            
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$r^2F(r,g)$")
    
    plt.errorbar(x_ac[0], r2F_r1_ac[0], **plot.data(2), label = "l=1")
    plt.errorbar(x_ac[1], r2F_r1_ac[1], **plot.data(3), label = "l=2")
    plt.errorbar(x_ac[2], r2F_r1_ac[2], **plot.data(4), label = "l=3")
    
    plt.legend()
    
    plt.savefig("/home/negro/projects/matching/step_scaling/confronto_r2F_r1_L4.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
def confronto_r2F_r2_L4(Ns):
    # [l=1, l=2, l=3]
    x_ac = [1/0.36915425472851615**2, 1/0.7531542547285165**2, 1/0.7400542547285165**2]
    r2F_r2_ac = [1.0226308779857438, 1.0248333693799925, 1.0655839649705927]
    
    betas = [1.6, 1.8, 2, 4, 4.05, 4.15, 4.25]

    plt.figure(figsize=(16,12))

    r = []
    r2F=[]
    d_r2F=[]
    for beta in betas:
        path = f"/home/negro/projects/matching/step_scaling/L{Ns}/T42_L{Ns}_b{beta}"
            
        tmp1, tmp2, tmp3 = np.loadtxt(f"{path}/analysis/r2F.txt", usecols=(0,1,2), unpack=True)
        r.append(tmp1)
        r2F.append(tmp2)
        d_r2F.append(tmp3)
    
    r = np.column_stack(r)
    r2F = np.column_stack(r2F)
    d_r2F = np.column_stack(d_r2F)
        
    plt.errorbar(betas, r2F[1, :], d_r2F[1, :], **plot.data(1), label='lagrangian')
        
    ## fitting 
    # linear model
    def model(x, a, b, c):
            return a + b*x +c*x**2
        
    # data to fit
    x_min = np.minimum(betas[0], np.min(x_ac))
    x_max = betas[-1]
    y = r2F[1, :]
    d_y = d_r2F[1, :]
        
    # starting parameters and bounds
    p0 = [1,1,1]
    bounds =  -np.inf, np.inf

    # fitting
    opt, cov = curve_fit(model, betas, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        
    # x_fit & y_fit
    x_fit = np.linspace(x_min, x_max, 100)
    y_fit = model(x_fit, opt[0], opt[1], opt[2])
    
    # plotting the fit curve
    plt.plot(x_fit, y_fit, **plot.fit(1), label = 'quadratic fit')
        
    ## confidence band
    # bootstrap samples
    n_boot = 200

    boot_opt = np.random.multivariate_normal(opt, cov, size=n_boot, tol=1e-10)
        
    boot_y_fit = [model(x_fit, boot_opt[i,0], boot_opt[i,1], boot_opt[i,2]) for i in range(n_boot)]        
    boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
    plt.fill_between(x_fit, y_fit - boot_band, y_fit + boot_band, **plot.conf_band(1))
            
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$r^2F(r,g)$")
    
    plt.errorbar(x_ac[0], r2F_r2_ac[0], **plot.data(2), label = "l=1")
    plt.errorbar(x_ac[1], r2F_r2_ac[1], **plot.data(3), label = "l=2")
    plt.errorbar(x_ac[2], r2F_r2_ac[2], **plot.data(4), label = "l=3")
    
    plt.legend()
    
    plt.savefig("/home/negro/projects/matching/step_scaling/confronto_r2F_r2_L4.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def confronto_r2F_L3():
    path = "/home/negro/projects/matching/step_scaling/L3/"

    x = [1.0, np.sqrt(5)]
    
    betas = [1.4, 1.]
    
    # [l=1, l=2, l=3, l=4]
    r2F_r1_ac = [0.29536698, 0.19739002867530875, 0.18957441018081542, 0.18940793972490294]
    r2F_r2_ac = [1.02175514, 1.0250710849945603, 1.0648107364285249, 1.065914289433689]
    
    plt.figure(figsize=(18,12))
    
    for i, y in enumerate(r2F_r1_ac):
        plt.errorbar(x[0], y, **plot.data(i+1, label = f"l={i+1}"))
    
    for i, y in enumerate(r2F_r2_ac):
        plt.errorbar(x[1], y, **plot.data(i+1))
        
    for i, beta in enumerate([1.8, 1.9, 2]):
        r2F, d_r2F = np.loadtxt(f"{path}/T42_L3_b{beta}/analysis/r2F.txt", usecols=(1,2), unpack=True)
        plt.errorbar(x, r2F, d_r2F, **plot.data(0), label = ("lagrangian" if i==0 else ""))
        
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.xlabel(r'$r$')
    plt.ylabel(fr'$r^2F(r,1/g^2)$')
    
    plt.legend()
    plt.savefig("/home/negro/projects/matching/step_scaling/matching_beta_Ns3.png", dpi=300, bbox_inches='tight')

    plt.show()

def plot_AC():
    
    beta_ac_Ns3 = [1.4]
    
    r2F_r1_ac_Ns3 = [0.29536698, 0.19739002867530875, 0.18957441018081542, 0.18940793972490294]
    r2F_r2_ac_Ns3 = [1.02175514, 1.0250710849945603, 1.0648107364285249, 1.065914289433689]
    
    beta_ac_Ns4 = [1/0.36915425472851615**2, 1/0.7531542547285165**2, 1/0.7400542547285165**2]

    r2F_r1_ac_Ns4 = [0.14874300045572017, 0.2400338548299101, 0.20420630542747306]
    r2F_r2_ac_Ns4 = [1.0226308779857438, 1.0248333693799925, 1.0655839649705927]
    
    plt.figure(figsize=(18,12))
    
    for i in [2]:
        plt.errorbar(beta_ac_Ns3, r2F_r1_ac_Ns3[i], **plot.data_nomarker(1), marker = "v" , label=fr"$r_1=1$, $N_s=3$")
        plt.errorbar(beta_ac_Ns3, r2F_r2_ac_Ns3[i], **plot.data_nomarker(1), marker = "^", label=fr"$r_2=\sqrt{5}$, $N_s=3$")
        plt.errorbar(beta_ac_Ns4[i], r2F_r1_ac_Ns4[i], **plot.data_nomarker(2), marker = "D", label=fr"$r_1=\sqrt{2}$, $N_s=4$")
        plt.errorbar(beta_ac_Ns4[i], r2F_r2_ac_Ns4[i], **plot.data_nomarker(2), marker = "o", label=fr"$r_2=\sqrt{10}$, $N_s=4$")

    plt.title(f"l={i+1}")   
        
    plt.xlabel(r"$l$")
    plt.ylabel(fr'$r^2F(r,1/g^2)$')
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
        
    plt.legend()
    plt.savefig(f"/home/negro/projects/matching/step_scaling/plots_AC/plot.png", dpi=300, bbox_inches='tight')

    plt.show()

def tuning_barecoup():
    path = "/home/negro/projects/matching/step_scaling/L3/"
    
    betas = [1.4, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975]
    
    # [l=1, l=2, l=3, l=4]
    r2F_r1_ac = [0.29536698, 0.19739002867530875, 0.18957441018081542, 0.18940793972490294]
    r2F_r2_ac = [1.02175514, 1.0250710849945603, 1.0648107364285249, 1.065914289433689]
    
    plt.figure(figsize=(18,12))

    plt.errorbar(betas[0], r2F_r1_ac[3], **plot.data(0), label = "hamiltonian, l=4")
    
    r2F_r1 = []
    d_r2F_r1 = []
    for i, beta in enumerate(betas[1:]):
        foo, bar = np.loadtxt(f"{path}/T48_L3_b{beta}/analysis/r2F.txt", usecols=(1,2), unpack=True)
        r2F_r1.append(foo[0])
        d_r2F_r1.append(bar[0])
    
    plt.errorbar(betas[1:], r2F_r1, d_r2F_r1, **plot.data(1), label = ("lagrangian" if i==0 else ""))
    
    def quadratic(x, a, b, c):
        return a + b*x + c*x**2
    
    opt, cov, x_fit, y_fit, boot_band, chi2 = reg.fit_with_scipy(betas[1:], r2F_r1, d_r2F_r1, quadratic, [1,1,1], mask=None)
    
    a, b, c = opt[0], opt[1], opt[2]
    y0 = r2F_r1_ac[3]

    discriminant = b**2 - 4 * c * (a - y0)
    if discriminant < 0:
        raise ValueError("No real solution — the curve never reaches this value.")

    x1 = (-b + np.sqrt(discriminant)) / (2 * c)
    x2 = (-b - np.sqrt(discriminant)) / (2 * c)
    
    plt.plot(x_fit, y_fit, **plot.fit(1), label="lagrangian")
    plt.fill_between(x_fit, y_fit-boot_band, y_fit+boot_band, **plot.conf_band(1))
    
    plt.plot([1.4, x2],[r2F_r1_ac[3], r2F_r1_ac[3]], **plot.fit(5))
    
    rng = np.random.default_rng(seed=8220)
    boot_opt_sp = rng.multivariate_normal(opt, cov, size=1000, tol=1e-10)
    
    boot_y_fit_sp = np.array([quadratic(x2, *params) for params in boot_opt_sp])
    err = np.std(boot_y_fit_sp, axis=0, ddof=1)
        
    plt.errorbar(x2, quadratic(x2, *opt), err, **plot.data(5))
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.xlabel(f"$1/g^2$")
    plt.ylabel(f"$r^2F(r_1,1/g^2)$")
        
    plt.legend()
    plt.savefig("/home/negro/projects/matching/step_scaling/tuning_barecoup_Ns3_r1.png", dpi=300, bbox_inches='tight')
    
    return x2

def deduce_runcoup_r2(x2):
    def fit_r2(x, a, b):
        return a + b*x
    
    path = "/home/negro/projects/matching/step_scaling/L3/"

    betas = [1.4, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975]
    r2F_r2_ac = 1.065914289433689
    
    r2F_r2 = []
    d_r2F_r2 = []
    for beta in betas[1:]:
        foo, bar = np.loadtxt(f"{path}/T48_L3_b{beta}/analysis/r2F.txt", usecols=(1,2), unpack=True)
        r2F_r2.append(foo[1])
        d_r2F_r2.append(bar[1])
        
    plt.figure(figsize = (18,12))

    plt.errorbar(betas[0], r2F_r2_ac, **plot.data(0), label="hamiltonian, l=4")
    
    plt.errorbar(betas[1:], r2F_r2, d_r2F_r2, **plot.data(1), label = "lagrangian" )
    
    opt, cov, x_fit, y_fit, boot_band, chi2red = reg.fit_with_scipy(betas[1:], r2F_r2, d_r2F_r2, fit_r2, [1,1], mask=None)
    
    plt.plot(x_fit, y_fit, **plot.fit(1), label = fr"$\chi^2_r$={chi2red:.2f}")
    plt.fill_between(x_fit, y_fit-boot_band, y_fit+boot_band, **plot.conf_band(1))
    
    rng = np.random.default_rng(seed=8220)
    boot_opt_sp = rng.multivariate_normal(opt, cov, size=1000, tol=1e-10)
    
    boot_y_fit_sp = np.array([fit_r2(x2, *params) for params in boot_opt_sp])
    err = np.std(boot_y_fit_sp, axis=0, ddof=1)
        
    plt.errorbar(x2, fit_r2(x2, *opt), err, **plot.data(5))
    
    plt.plot([1.4, x2],[fit_r2(x2, *opt), fit_r2(x2, *opt)], **plot.fit(5))
    plt.fill_between(np.linspace(1.4, x2, 100),fit_r2(x2, *opt)-err, fit_r2(x2, *opt)+err, **plot.conf_band(5))
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.xlabel(f"$1/g^2$")
    plt.ylabel(f"$r^2F(r_2,1/g^2)$")
        
    plt.legend()
    
    plt.savefig("/home/negro/projects/matching/step_scaling/deduce_barecoup_Ns3_r2.png", dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    ## basic analysis ####
    for beta in [1.4]:
        path_glob = f"/home/negro/projects/matching/step_scaling/L3/T48_L3_b{beta}"
            
        #thermalization(path_glob)
        
        #concatenate.concatenate(f"{path_glob}/data", 10000, f"{path_glob}/analysis")
            
        #blocksize_analysis_primary(path)
        #blocksize_analysis_secondary(path)
        
        wsplot = [1, np.sqrt(5), np.sqrt(8)]
        #wsplot = [np.sqrt(2), np.sqrt(10), np.sqrt(18)]
        #wsplot = [np.sqrt(5), np.sqrt(25), np.sqrt(32)]
        #wsplot = [np.sqrt(8), np.sqrt(40), np.sqrt(72)]
        
        wtmax = 10
        
        #get_potential_wt(path_glob, wsplot, wtmax)
        
        #plot_effmass(path_glob, wtmax)        
        
        min_list = [2, 3, 4]
        max_list = [10, 10, 10]
        
        #fit_potential_ws(path_glob, wsplot, min_list, max_list)
        
        #compute_r2F(path_glob, wsplot)  
    
    ######################
    
    ## second stage analysis ####
    
    #tune_r2F()

    #path = "/home/negro/projects/matching/step_scaling/tune_b3"
    #plot_r2F_vs_rlatt(path)
    
    #confronto_r2F_r1_L4(4)
    #confronto_r2F_r2_L4(4)

    #confronto_r2F_L3()
    x2 = tuning_barecoup()
    deduce_runcoup_r2(x2)
    #plot_AC()
    
    #############################