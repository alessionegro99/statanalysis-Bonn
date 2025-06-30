import sys
import os
import shutil

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import concatenate
import plot
import progressbar as pb
import bootstrap as boot
import regression as reg

def readfile(path):
    output = np.loadtxt(f'{path}/analysis/dati.dat', skiprows=1)
    columns = [output[:, i] for i in range(output.shape[1])]

    # plaqs plaqt W(wt=1,ws=1) ... W(wt=wtmax,ws=1) W(wt=1,ws=sqrt(5)) ... W(wt=wtmax,ws=sqrt(5)) W(wt=1,ws=sqrt(8)) ... W(wt=wtmax,ws=sqrt(8))
    
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
 
def plot_thermalization(path):
    x,y = np.loadtxt(f"{path}/analysis/thermalization.txt", )
        
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
        plt.errorbar(np.arange(1,wtmax[i]+1), effma[:wtmax[i]], d_effma[:wtmax[i]], **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
                
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/effmass.png", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$')

    for i, (effma, d_effma) in enumerate(zip(np.transpose(effm), np.transpose(d_effm))):
        plt.errorbar(1/np.arange(1,wtmax[i]+1), effma[:wtmax[i]], d_effma[:wtmax[i]], **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
    
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/potential_ws.png", dpi=300, bbox_inches='tight')

def find_tmin(path, wtmax, tmin, tmax):
    def model(x, a, b):
        return a + b/x
    
    data = np.load(f"{path}/analysis/potential_wt.npy", allow_pickle=True)
    
    _, pot, d_pot, pot_bs = map(np.array, data[:4])
    
    bounds = -np.inf, np.inf
    
    p0 = [1, 1]
    
    V0 = []
    chi2red = []
    
    for i, (pota, d_pota) in enumerate(zip(np.transpose(pot), np.transpose(d_pot))):
        wt = np.arange(1, wtmax[i]+ 1)
        
        min = tmin[i]
        max = tmax[i]
        
        x = wt[min:max]
        y = pota[min:max]
        d_y = d_pota[min:max]
        
        opt, cov = curve_fit(model, x, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        chi2 = reg.chi2_corr(x, y, model, np.diag(d_y**2), opt[0], opt[1])
        chi2red.append(chi2/(len(x) - len(opt)))
        V0.append(opt[0])

    p0 = opt
    
    d_V0 = []
    for i in range(pot_bs.shape[1]):
        wt = np.arange(1, wtmax[i]+1)
        pota_bs = pot_bs[:,i,:]
        
        d_pota = np.std(pota_bs, axis=1, ddof=1)
        
        opt_bs = []        
        
        for j in range(pota_bs.shape[1]):
            pota = pota_bs[:,j]
            
            min = tmin[i]
            max = tmax[i]
            
            x = wt[min:max]
            y = pota[min:max]
            d_y = d_pota[min:max]
            
            opt, _ = curve_fit(model, x, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
            
            opt_bs.append(opt[0])        
        d_V0.append(np.std(opt_bs[:]))
    
    return tmin, tmax, V0, d_V0, chi2red   
    
def plot_find_tmin(path):
        tminprobe = []
        Vprobe = []
        d_Vprobe = []
        chi2redprobe = []
        
        for a, b, c in zip(range(2,20), range(2,20), range(2,20)):
            pb.progress_bar(a, 20)
            tmax = [24, 24, 24]
            tmin = [a, b, c]     
            foo, _, bar, gar, goo = find_tmin(path_glob, wtmax, tmin, tmax)
            tminprobe.append(np.array(foo))
            Vprobe.append(np.array(bar))
            d_Vprobe.append(np.array(gar))
            chi2redprobe.append(np.array(goo))
            
        tminprobe = np.transpose(tminprobe)
        Vprobe = np.transpose(Vprobe)
        d_Vprobe = np.transpose(d_Vprobe)
        chi2redprobe = np.transpose(chi2redprobe)
                
        for i in [0, 1, 2]:
            plt.figure(figsize=(18,12))
            plt.xlabel(r'$t_{min}$')
            plt.ylabel(r'$aV(w_t)$')
            x = tminprobe[i]
            y = Vprobe[i]
            d_y = d_Vprobe[i]
            plt.errorbar(x, y, d_y, **plot.data(i))
            chi2=chi2redprobe[i]
            for xi, yi, chi2i in zip(x, y, chi2):
                plt.text(xi, yi - 0.1 * max(d_y), f"{chi2i:.2f}", 
                        ha='center', va='top', fontsize=20, rotation=45)
            plt.grid (True, linestyle = '--', linewidth = 0.25)
            plt.savefig(f"{path_glob}/analysis/V0_vs_tmin_{wsplot[i]:.2f}.png", dpi=300, bbox_inches='tight')     
    
def boot_fit_potential_ws(path, wtmax, mfit_lst, Mfit_lst):
    
    def model(x, a, b):
        return a + b/x
    
    def linearized_model(z, A, B):
        return A + B*z
    
    data = np.load(f"{path}/analysis/potential_wt.npy", allow_pickle=True)
    
    wsplot, pot, d_pot, pot_bs = map(np.array, data[:4])
        
    plt.figure(figsize=(18,12))
    
    plt.xlabel(r'$w_t$')
    plt.ylabel(r'$aV(w_t)$')
    
    bounds = -np.inf, np.inf
    
    p0 = [1, 1]
    
    V0 = []
    print("original sample")
    print("V0 d_V0 chi2red")
    # fit on the original sample    

    z_fit = []
    y_fit = []

    for i, (pota, d_pota) in enumerate(zip(np.transpose(pot), np.transpose(d_pot))):
        wt = np.arange(1, wtmax[i]+ 1)
        
        min = mfit_lst[i]
        max = Mfit_lst[i]
        
        x = wt[min:max]
        y = pota[min:max]
        d_y = d_pota[min:max]
        
        opt, cov = curve_fit(model, x, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
        chi2 = reg.chi2_corr(x, y, model, np.diag(d_y**2), opt[0], opt[1])
        chi2red = chi2/(len(x) - len(opt))
        
        print(opt[0], cov[0][0]**0.5, chi2red)
        V0.append(opt[0])
        
        z_fit.append(np.linspace(0, 1/x[0], 200))
        y_fit.append(linearized_model(z_fit[i], opt[0], opt[1]))
        
        plt.errorbar(1/wt, pota[:wtmax[i]], d_pota[:wtmax[i]], **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$, $\chi^2/dof$={chi2red:.2f}")

    p0 = opt
    
    print("bootstrapping the fit")
    print("boot_V0 d_boot_V0 std_cov")
    # bootstrapping the fit
    boot_V0 = []
    for i in range(pot_bs.shape[1]):
        wt = np.arange(1, wtmax[i]+1)
        pota_bs = pot_bs[:,i,:]
        
        d_pota = np.std(pota_bs, axis=1, ddof=1)
        
        opt_bs = []
        cov_bs = []
        
        boot_y_fit = []
        
        for j in range(pota_bs.shape[1]):
            pb.progress_bar(j,pota_bs.shape[1])
            pota = pota_bs[:,j]
            #plt.errorbar(1/wt, pota[:wtmax[i]], d_pota[:wtmax[i]], **plot.data(i), label=fr"$w_s={wsplot[i]:.2f}$")
            
            min = mfit_lst[i]
            max = Mfit_lst[i]
            
            x = wt[min:max]
            y = pota[min:max]
            d_y = d_pota[min:max]
            
            opt, cov = curve_fit(model, x, y, sigma=d_y, absolute_sigma=True, p0=p0, bounds=bounds)
            
            opt_bs.append(opt[0])
            cov_bs.append(cov[0][0]**0.5)
            
            boot_y_fit.append(linearized_model(z_fit[i], opt[0], opt[1]))
        boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
        
        plt.fill_between(z_fit[i], y_fit[i] - boot_band, y_fit[i] + boot_band, **plot.conf_band(i))
        plt.plot(z_fit[i], y_fit[i], **plot.fit(i))
        
        print(np.mean(opt_bs[:]), np.std(opt_bs[:]), np.mean(cov_bs[:]))
        boot_V0.append(np.array(opt_bs))
    
    plt.xlim(-0.01,0.3)

    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/analysis/boot_extrap_potential_ws.png", dpi=300, bbox_inches='tight')    

    np.save(f"{path}/analysis/opt", np.array([V0, boot_V0], dtype=object))
   
def compute_r2F(path, wsplot):    
    opt = np.load(f"{path}/analysis/opt.npy", allow_pickle=True)
    
    pot = opt[0]
    boot_pot = opt[1]
    
    r2F_r1 = wsplot[0]**2*(pot[1]-pot[0])/(wsplot[1]-wsplot[0])
    r2F_r2 = wsplot[1]**2*(pot[2]-pot[1])/(wsplot[2]-wsplot[1])
    
    boot_r2F_r1 = wsplot[0]**2*(boot_pot[1]-boot_pot[0])/(wsplot[1]-wsplot[0])
    boot_r2F_r2 = wsplot[1]**2*(boot_pot[2]-boot_pot[1])/(wsplot[2]-wsplot[1])
    
    d_r2F_r1 = np.std(boot_r2F_r1, ddof=1)
    d_r2F_r2 = np.std(boot_r2F_r2, ddof=1)
    
    data = np.column_stack((np.array([wsplot[0], wsplot[1]]), np.array([r2F_r1, r2F_r2]), np.array([d_r2F_r1, d_r2F_r2])))
    
    np.savetxt(f"{path}/analysis/r2F.txt", data)
        
def tune_r2F_r2(path):
    r2F_r1, r2F_r2 = [], []
    
    d_r2F_r1, d_r2F_r2 = [], []    
    Ns_lst = [3, 4, 5, 7]
    beta_lst = [[3], [4, 4.05, 4.1, 4.15, 4.2, 4.25], [11, 11.5, 12, 12.5], [15]]

    for i, Ns in enumerate(Ns_lst):
        foo, d_foo = [], []
        goo, d_goo = [], []

        for beta in beta_lst[i]:
            with open(f"{path}/L{Ns}/T48_L{Ns}_b{beta}/analysis/r2F.txt", "r") as file:
                # Read first line (r1)
                line1 = next(file)
                y1 = float(line1.split()[1])
                dy1 = float(line1.split()[2])

                # Read second line (r2)
                line2 = next(file)
                y2 = float(line2.split()[1])
                dy2 = float(line2.split()[2])

                foo.append(y1)
                d_foo.append(dy1)
                goo.append(y2)
                d_goo.append(dy2)

        # Append per-Ns group
        r2F_r1.append(foo)
        d_r2F_r1.append(d_foo)
        r2F_r2.append(goo)
        d_r2F_r2.append(d_goo)

    # fitting r2F_r2
    plt.figure(figsize=(18,12))
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$r^2F(r_2,g)$")
    for i, Ns in enumerate(Ns_lst):
        x, y, d_y = beta_lst[i], r2F_r2[i], d_r2F_r2[i]
        if i == 0 or i == 3:
            plt.errorbar(x, y, d_y, **plot.data(i), label = fr"$N_s$={i}")
        else:
            bounds = -np.inf, np.inf
            
            def linear(x, q, m):
                return q + m*x
            
            p0 = [1,-1]
            
            opt, cov = curve_fit(linear, x, y, sigma = d_y, absolute_sigma=True, p0=p0, bounds=bounds)
            
            chi2red = reg.chi2_corr(np.array(x), y, linear, np.diag(np.array(d_y)**2), opt[0], opt[1])/(len(x)-len(opt))

            x_fit = np.linspace(x[0], x[-1], 100)
            y_fit = linear(x_fit, *opt)
                    
            rng = np.random.default_rng(seed=8220)  
            boot_opt = rng.multivariate_normal(opt, cov, size=500, tol=1e-10)
                        
            boot_y_fit = [linear(x_fit, boot_opt[i,0], boot_opt[i,1]) for i in range(200)]
            boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
            
            plt.errorbar(x, y, d_y, **plot.data(i), label = fr"$N_s$={i}")
            plt.plot(x_fit, y_fit, **plot.fit(i), label = fr"$\chi^2$={chi2red:.2f}")
            plt.fill_between(x_fit, y_fit-boot_band, y_fit+boot_band, **plot.conf_band(i))
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/tuning_b3/tuning_b3_r2.png", dpi=300, bbox_inches='tight')    

    # fitting r2F_r1
    plt.figure(figsize=(18,12))
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$r^2F(r_1,g)$")
    for i, Ns in enumerate(Ns_lst):
        x, y, d_y = beta_lst[i], r2F_r1[i], d_r2F_r1[i]
        if i == 0 or i == 3:
            plt.errorbar(x, y, d_y, **plot.data(i), label = fr"$N_s$={i}")
        else:
            bounds = -np.inf, np.inf
            
            def linear(x, q, m):
                return q + m*x
            
            p0 = [1,-1]
            
            opt, cov = curve_fit(linear, x, y, sigma = d_y, absolute_sigma=True, p0=p0, bounds=bounds)
            
            chi2red = reg.chi2_corr(np.array(x), y, linear, np.diag(np.array(d_y)**2), opt[0], opt[1])/(len(x)-len(opt))

            x_fit = np.linspace(x[0], x[-1], 100)
            y_fit = linear(x_fit, *opt)
                    
            rng = np.random.default_rng(seed=8220)  
            boot_opt = rng.multivariate_normal(opt, cov, size=500, tol=1e-10)
                        
            boot_y_fit = [linear(x_fit, boot_opt[i,0], boot_opt[i,1]) for i in range(200)]
            boot_band = np.std(boot_y_fit, axis = 0, ddof=1)   
            
            plt.errorbar(x, y, d_y, **plot.data(i), label = fr"$N_s$={i}")
            plt.plot(x_fit, y_fit, **plot.fit(i), label = fr"$\chi^2$={chi2red:.2f}")
            plt.fill_between(x_fit, y_fit-boot_band, y_fit+boot_band, **plot.conf_band(i))
    plt.grid (True, linestyle = '--', linewidth = 0.25)
    plt.legend()
    
    plt.savefig(f"{path}/tuning_b3/tuning_b3_r1.png", dpi=300, bbox_inches='tight')    

    plt.show()  

if __name__ == "__main__":
    ## basic analysis ####
    Ns = 7
    for beta in [15]:
        path_glob = f"/home/negro/projects/matching/step_scaling/L{Ns}/T48_L{Ns}_b{beta}"
                                
        #plot_thermalization(path_glob)
        
        #blocksize_analysis_primary(path)
        #blocksize_analysis_secondary(path)
        
        if Ns==3:
            wsplot = [1, np.sqrt(5), np.sqrt(8)]
        elif Ns==4:
            wsplot = [np.sqrt(2), np.sqrt(10), np.sqrt(18)]
        elif Ns==5:
            wsplot = [np.sqrt(5), np.sqrt(25), np.sqrt(32)]
        elif Ns==7:
            wsplot = [np.sqrt(8), np.sqrt(40), np.sqrt(72)]
        
        wtmax = [24, 24, 24]
                
        # plot_effmass(path_glob, wtmax)
        
        # plot_find_tmin(path_glob)
            
        mfit_lst = [4, 8, 8]
        Mfit_lst = [24, 24, 24]
        
        #boot_fit_potential_ws(path_glob, wtmax, mfit_lst, Mfit_lst)

        #compute_r2F(path_glob, wsplot)   
    
    ## second stage 
    
    path_ss = f"/home/negro/projects/matching/step_scaling"
    
    tune_r2F_r2(path_ss)

        