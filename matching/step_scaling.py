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

    # plaqs plaqt W(wt=1,ws=1) ... W(wt=10,ws=1) W(wt=1,ws=sqrt(5)) ... W(wt=10,ws=sqrt(5)) W(wt=10,ws=1) W(wt=1,ws=sqrt(8)) ... W(wt=10,ws=sqrt(8))
    
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
    plt.title(r'MC history of temporal plaquette $U_t$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)

    plt.savefig(f"{path}/analysis/thermalization.png", dpi=300, bbox_inches='tight')
        
def blocksize_analysis_primary(path):
    print("reading file...")
    data = readfile(path)
    print("read file ")
    
    for aux in [0, 1]:
        obs=data[:,aux]
        boot.blocksize_analysis_primary(obs, 200, [10, 500, 5], savefig=1, path=f"{path}/analysis/")
 
def blocksize_analysis_secondary(path):
    print("reading file...")
    data = readfile(path)
    print("done.")
    
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

def plot_potential_wt(path, savefig=0):
    data = readfile(path)
    
    def id(x):
        return x
    
    seed = 8220
    samples = 200
    blocksizes = [50, 50, 50]
    
    wtmax = 10
    wsmax = 3
    
    wtmaxplot = 10
    
    #wsplot = [1, np.sqrt(5), np.sqrt(8)]
    #wsplot = [np.sqrt(2), np.sqrt(10), np.sqrt(18)]
    #wsplot = [np.sqrt(5), np.sqrt(25), np.sqrt(32)]
    wsplot = [np.sqrt(8), np.sqrt(40), np.sqrt(72)]
    
    plt.figure(figsize=(16,12))    
    for wt in range(wtmaxplot):
        V = []
        d_V = []
        
        for ws, blocksize in zip(range(wsmax), blocksizes):
            W = data[:, 2 + wt + wtmax*ws]

            args = [id, W]
            
            def potential(x):
                eps=1e-10
                return -np.log(np.clip(x, eps, None))/(wt+1)
            
            print(f"Currently bootstrapping w_t={wt+1}, w_s={wsplot[ws]:.2f}...")
            ris, err = boot.bootstrap_for_secondary(potential, blocksize, samples, 1, args, seed=seed)
            
            V.append(potential(np.mean(W)))
            d_V.append(err)
        
        sys.stdout.flush()

        plt.errorbar(wsplot, V, d_V, **plot.data(wt), label=fr'$w_t={wt+1}$')

        tofile = np.column_stack((np.array(wsplot), np.array(V), np.array(d_V)))

        np.savetxt(f"{path}/analysis/potential_wt/potential_wt_{wt+1}.txt", tofile)
        
    plt.xlabel(r'$w_s$')
    plt.ylabel(r'$aV(w_s)$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    if savefig==0:
        plt.show()
    elif savefig==1:
        plt.savefig(f"{path}/analysis/potential_wt.png", dpi=300, bbox_inches='tight')
        
def plot_potential_ws(path, savefig=0):
    wsmax = 3
    
    wtmaxplot = 10
    
    #wsplot = [1, np.sqrt(5), np.sqrt(8)]
    #wsplot = [np.sqrt(2), np.sqrt(10), np.sqrt(18)]
    #wsplot = [np.sqrt(5), np.sqrt(25), np.sqrt(32)]
    wsplot = [np.sqrt(8), np.sqrt(40), np.sqrt(72)]
        
    plt.figure(figsize=(16,12))    
    for ws in range(wsmax):
        V = []
        d_V = []
        
        for wt in range(wtmaxplot):
            V_tmp, d_V_tmp = np.loadtxt(f"{path}/analysis/potential_wt/potential_wt_{wt+1}.txt", usecols=(1,2), unpack=True)

            V.append(V_tmp[ws])
            d_V.append(d_V_tmp[ws])
        
        plt.errorbar(1/np.arange(1,wtmaxplot+1), V, d_V, **plot.data(ws), label=fr'$w_s={wsplot[ws]:.2f}$')
        
    plt.xlabel(r'$1/w_t$')
    plt.ylabel(r'$aV(w_t)$', rotation=0)
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    plt.title(r'$aV(w_t, w_s)$')
            
    plt.xticks(rotation=0)  
    plt.yticks(rotation=0) 
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    if savefig==0:
        plt.show()
    elif savefig==1:
        plt.savefig(f"{path}/analysis/potential_ws.png", dpi=300, bbox_inches='tight')

def plot_fit_potential_ws(path, ws, xmin, xmax):
    # for fitting aV(wt,ws) for small wt
    def ansatz(x, pars):
        return pars[0] + pars[1]/x
    
    # samples for bootstrapping
    samples = 500

    # initial guess of parameters
    params = [1,1]         
    
    # max wt used for extrapolating
    wtmaxplot = 10
        
    #wsplot = [1, np.sqrt(5), np.sqrt(8)]
    #wsplot = [np.sqrt(2), np.sqrt(10), np.sqrt(18)]
    #wsplot = [np.sqrt(5), np.sqrt(25), np.sqrt(32)]
    wsplot = [np.sqrt(8), np.sqrt(40), np.sqrt(72)]

    # stuff that gets printed to file
    V_0_file = []
    d_V_0_file = []
    b_file = []
    d_b_file = []
    chi2red_file = []

    y_t0 = []
    d_y = []
                
    for wt in range(wtmaxplot):
        V_tmp, d_V_tmp = np.loadtxt(f"{path}/analysis/potential_wt/potential_wt_{wt+1}.txt", usecols=(1,2), unpack=True)
            
        y_t0.append(V_tmp[ws]) #
        d_y.append(d_V_tmp[ws])
        
    x = np.arange(1, wtmaxplot+1)
    y_t0 = np.array(y_t0)
    d_y = np.array(d_y)           
    
    ris, err, chi2, dof, pvalue, boot_sample = reg.fit_with_yerr(x, y_t0, d_y, xmin, xmax, ansatz, params, samples \
                                                                    , save_figs=1, save_path=f'{path}/analysis/extrapolation/ws_{wsplot[ws]:.2f}'\
                                                                    , plot_title = fr'$aV(w_t,w_s={wsplot[ws]:.2f})$', xlab = r'$w_t$', ylab = r'$aV(w_t)$')
         
    print(ris, err, chi2/dof)
           
    V_0_file.append(ris[0])
    d_V_0_file.append(err[0])
    b_file.append(ris[1])
    d_b_file.append(err[1])
    chi2red_file.append(chi2/dof)        
        
    data = np.column_stack((wsplot[ws], np.array(V_0_file), np.array(d_V_0_file), np.array(b_file), np.array(d_b_file), np.array(chi2red_file)))
    
    np.savetxt(f"{path}/analysis/extrapolation/ws_{wsplot[ws]:.2f}/fit_results.txt", data)
    np.savetxt(f"{path}/analysis/extrapolation/ws_{wsplot[ws]:.2f}/fit_results_boot_samples.txt", boot_sample)

def compute_r2F(path):
    wsplot = [1, np.sqrt(5), np.sqrt(8)]
    #wsplot = [np.sqrt(2), np.sqrt(10), np.sqrt(18)]
    #wsplot = [np.sqrt(5), np.sqrt(25), np.sqrt(32)]
    #wsplot = [np.sqrt(8), np.sqrt(40), np.sqrt(72)]

    V_r = []
    boot_V_r = []
    for ws in wsplot:
        tmp = np.loadtxt(f"{path}/analysis/extrapolation/ws_{ws:.2f}/fit_results.txt", usecols=(1), unpack=True)
        V_r.append(tmp)
        
        tmp = np.loadtxt(f"{path}/analysis/extrapolation/ws_{ws:.2f}/fit_results_boot_samples.txt", unpack=True)
        boot_V_r.append(tmp[:,0])

    boot_V_r = np.column_stack(boot_V_r)

    r2F_r1 = wsplot[0]**2*(V_r[1]-V_r[0])/(wsplot[1]-wsplot[0])
    r2F_r2 = wsplot[1]**2*(V_r[2]-V_r[1])/(wsplot[2]-wsplot[1])
    
    boot_r2F_r1 = wsplot[0]**2*(boot_V_r[:,1]-boot_V_r[:,0])/(wsplot[1]-wsplot[0])
    boot_r2F_r2 = wsplot[1]**2*(boot_V_r[:,2]-boot_V_r[:,1])/(wsplot[2]-wsplot[1])
    
    d_r2F_r1 = np.std(boot_r2F_r1, ddof=1)
    d_r2F_r2 = np.std(boot_r2F_r2, ddof=1)
    
    data = np.column_stack((np.array([wsplot[0], wsplot[1]]), np.array([r2F_r1, r2F_r2]), np.array([d_r2F_r1, d_r2F_r2])))
    
    np.savetxt(f"{path}/analysis/r2F.txt", data)
    
def tune_r2F():
    Ns_list=[4, 5]
    
    stuff = []
    runcoup_r1 = []
    d_runcoup_r1 = []
    
    betas_list = [[4, 4.05, 4.15, 4.25], [11, 11.5, 12, 12.5]]

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


if __name__ == "__main__":
    path = "/home/negro/projects/matching/step_scaling/L3/T42_L3_b3"
    
    #concatenate.concatenate(f"{path}/data", 100)
    
    #thermalization_plaqt(path)
    
    #blocksize_analysis_primary(path)
    #blocksize_analysis_secondary(path)
    
    #plot_potential_wt(path, 1)
    
    #plot_potential_ws(path, 1)
    #plot_fit_potential_ws(path, 2, 7, 10)
    
    #compute_r2F(path)  
    #tune_r2F()

    path = "/home/negro/projects/matching/step_scaling/tune_b3"
    #plot_r2F_vs_rlatt(path)
    
    
    