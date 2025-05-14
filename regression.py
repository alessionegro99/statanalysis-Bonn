#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats

from scipy.optimize import curve_fit

import progressbar as pb
import plot

__all__ = ["fit_with_yerr", "fit_with_xyerr"]


def _residuals_yerr(params, x, y, dy, func):
   """
   Return the vector of residuals (normalized with the error)
   for func(x, param).
   """

   ris=(y-func(x, params) )/dy
   return ris


def fit_with_yerr(x, y, dy, xmin, xmax, func, params, samples, \
   stop_param=1.0e-15, plot_fit=1, plot_fit_total_range=0, plot_band=1, \
  plot_residuals=1, plot_distribution=1, save_figs=0, save_path = 'null', show_progressbar=1, plot_title='myplot', xlab='x', ylab='y'):
   """
   Perform a fit to data on [xmin, xmax] with the function func(x, param),
   using "samples" bootstrap samples to evaluate the errors.

   stop_param: stopping parameter for the least square regression
   plot_fit: if =1 plot the optimal fit together with the data
   plot_band: if =1 plot the 1std band together with data
   plot_residuals: if =1 plot residuals after convergence
   plot_distribtion: if =1 plot the bootstrapped distributions of the parameters
   save_figs: if =1 save the figures in png insted of displaying them
   show_progressbar: if =1 show the progressbar

   return the optimal vales of the parameters, their errors, 
   the value of chi^2,the number of dof, the p-value
   and the bootstrap samples of the parameters.
   """
   
   mask = ((x<=xmax) & (x>=xmin))
   x_total_range=x
   xmintr = x_total_range[0]
   xmaxtr = x_total_range[-1]
   x=x[mask]
   y=y[mask]
   dy=dy[mask]

   band_size=1000

   data_length=len(x)
 
   # array to store the bootstrapped results 
   boot_sample=np.empty((len(params), samples), dtype=np.float64)

   if plot_band==1:
     x_band=np.linspace(xmin, xmax, band_size)
     boot_band=np.empty((band_size, samples), dtype=np.float64)
     
     x_band_tr=np.linspace(xmintr, xmaxtr, band_size)
     boot_band_tr=np.empty((band_size, samples), dtype=np.float64)
     
   for i in range(samples): 
     if show_progressbar==1:
       pb.progress_bar(i, samples)

     # bootstrap sample
     booty=y+np.random.normal(0, dy, data_length) 

     # least square regression
     ris = opt.leastsq(_residuals_yerr, params, ftol=stop_param, args=(x, booty, dy, func))
     boot_sample[:,i]=ris[0]

     if plot_band==1:
       boot_band[:,i]=func(x_band, ris[0])
     
     if plot_band==1 and plot_fit_total_range==1: 
       boot_band_tr[:,i]=func(x_band_tr, ris[0])


   # optimal parameters and errors
   ris=np.mean(boot_sample, axis=1)
   err=np.std(boot_sample, axis=1, ddof=1)

   # auxilliary stuff
   opt_res=_residuals_yerr(ris, x, y, dy, func)
   chi2=np.sum(opt_res*opt_res)
   dof=data_length - len(params)
   pvalue=1.0 - stats.chi2.cdf(chi2, dof)


   if plot_fit==1:
     x_aux=np.linspace(xmin, xmax, 1000)
     y_aux=func(x_aux, ris)

     plt.figure('Best fit (chi2/dof=%.4f/%d=%f)' % (chi2, dof, chi2/dof), figsize = (16,12))
     
     plt.xticks(rotation=0)  
     plt.yticks(rotation=0) 
     
     plt.xlabel(xlab)
     plt.ylabel(ylab, rotation=0)
     plt.gca().yaxis.set_label_coords(-0.1, 0.5)
     plt.title(plot_title)
     
     plt.xlim(0.9*xmin, 1.1*xmax)
     plt.errorbar(x, y, yerr=dy, **plot.data(1))
     plt.plot(x_aux,y_aux,**plot.fit(1))
      
     if plot_band==1:
       band_mean=np.mean(boot_band, axis=1)
       band_std=np.std(boot_band, axis=1)
       plt.fill_between(x_band, band_mean - band_std, band_mean + band_std, **plot.conf_band(1))
      
     plt.xticks(rotation=0)
     plt.yticks(rotation=0)
     
     plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    
     if save_figs==1:
       plt.savefig(f'{save_path}/fit.png', dpi=300, bbox_inches='tight')
     else:
       plt.show()

   if plot_fit_total_range==1:
     x_aux=np.linspace(xmintr, xmaxtr, 1000)
     y_aux=func(x_aux, ris)

     plt.figure('Best fit (chi2/dof=%.4f/%d=%f)' % (chi2, dof, chi2/dof), figsize = (16,12))
     plt.xlim(0.9*xmintr, 1.1*xmaxtr)
     plt.errorbar(x, y, yerr=dy, **plot.data(1))
     plt.plot(x_aux,y_aux,**plot.fit(1))
      
     if plot_band==1:
       band_mean_tr=np.mean(boot_band_tr, axis=1)
       band_std_tr=np.std(boot_band_tr, axis=1)
       plt.fill_between(x_band_tr, band_mean_tr - band_std_tr, band_mean_tr + band_std_tr, **plot.conf_band(1))
      
     plt.xticks(rotation=0)
     plt.yticks(rotation=0)
     
     plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    
     if save_figs==1:
       plt.savefig(f'{save_path}/fit.png', dpi=300, bbox_inches='tight')
     else:
       plt.show()

   if plot_residuals==1:
     x_aux=np.linspace(xmin, xmax, 1000)
     y_aux=np.ones(len(x_aux))

     plt.figure('Residuals', figsize=(16,12))
     plt.grid(True, which='both', linestyle='--', linewidth=0.25)
     plt.xlim(0.9*xmin, 1.1*xmax)
     plt.errorbar(x, opt_res, yerr=1, **plot.data(1))
     plt.plot(x_aux, -2*y_aux, 'g:')
     plt.plot(x_aux, -y_aux, 'r--')
     plt.plot(x_aux, 0*y_aux, 'r-')
     plt.plot(x_aux, y_aux, 'r--')
     plt.plot(x_aux, 2*y_aux, 'g:')
     
     if save_figs==1:
       plt.savefig(f'{save_path}/residuals.png')
     else:
       plt.show()

   if plot_distribution==1:
     for i in range(len(params)):
       plt.figure('Bootstrapped distribution of param[%d]' % i, figsize=(16,12))
       plt.grid(True, which='both', linestyle='--', linewidth=0.25)
       plt.xlabel('param[%d] values' % i)
       plt.ylabel('distribution histogram')
       plt.hist(boot_sample[i], bins='auto')
       
       if save_figs==1:
         plt.savefig(f'{save_path}/param'+str(i)+'.png')
       else:
         plt.show()

   return ris, err, chi2, dof, pvalue, boot_sample

 ## to do: modify and use this only for the correlated fits
def fit_yerr_uncorrelated(func, x, y, d_y, bsamples, \
                          maskfit, maskplot, rangeplot, plotabstract=0, \
                            kwargs1=None, kwargs2=None, kwargs3=None, kwargs4=None, label=None):
  # masking for fitting
  xfit=x[maskfit[0]:maskfit[1]]
  yfit=y[maskfit[0]:maskfit[1]]
  d_yfit=d_y[maskfit[0]:maskfit[1]]
  bsamplesfit=bsamples[:,maskfit[0]:maskfit[1]]
  
  # masking for plotting
  xplot=x[maskplot[0]:maskplot[1]]
  yplot=y[maskplot[0]:maskplot[1]]
  d_yplot=d_y[maskplot[0]:maskplot[1]]
  bsamplesplot=bsamples[:,maskplot[0]:maskplot[1]]
  
  # performing uncorrelated fit on the original data
  opt, cov = curve_fit(func, xfit, yfit, sigma=d_yfit, absolute_sigma=True, **kwargs1)
  
  # computing the reduced chi2 on the original data
  residuals = yfit - func(xfit, *opt)
  chi2red = np.sum((residuals/d_yfit)**2)/(len(xfit) - len(opt))
  
  bandsize = 1000
  xband = np.linspace(rangeplot[0], rangeplot[-1], bandsize)
  
  boot_opt = []
  boot_cov = []
  boot_band = []
  boot_chi2red = []
  
  # bootstrapping the fit
  for sample in range(np.shape(bsamplesplot)[0]):
    aux = bsamplesfit[sample,:]
    
    aux1, aux2 = curve_fit(func, xfit, aux, sigma=d_yfit, absolute_sigma=True)
    
    boot_opt.append(aux1)
    boot_cov.append(aux2)
    boot_band.append(func(xband, *aux1))
    
  band_mean = np.mean(boot_band, axis = 0)
  band_std = np.std(boot_band, axis = 0)
  
  if plotabstract==0:
    plt.figure(figsize=(16,12))

  plt.errorbar(xplot, yplot, d_yplot, **kwargs2)
  plt.plot(xband, func(xband, *opt), **kwargs3, label=label)
  plt.fill_between(xband, band_mean-band_std, band_mean+band_std, **kwargs4)

  if plotabstract==0:
    plt.show()
  
  return opt, cov, chi2red, boot_opt, boot_cov
  

def _residuals_xyerr(extended_params, x, dx, y, dy, true_param_length, func):
   """
   Return the vector of residuals (normalized with the error)
   for func(x, param).
   
   extended_param[:true_param_length] = real parameters
   extended_param[true_param_length:] = auxilliary parameters like x
   """

   risy=(y-func(extended_params[true_param_length:], extended_params[:true_param_length]) )/dy
   risx=(x-extended_params[true_param_length:])/dx
   return np.append(risy, risx)


def fit_with_xyerr(x, dx, y, dy, xmin, xmax, func, params, samples, \
   stop_param=1.0e-15, plot_fit=1, plot_band=1, plot_residuals=1, \
   plot_distribution=1, save_figs=0, show_progressbar=1):
   """
   Perform a fit to data on [xmin, xmax] with the function func(x, param),
   using "samples" bootstrap samples to evaluate the errors.

   stop_param: stopping parameter for the least square regression
   plot_fit: if =1 plot the optimal fit together with the data
   plot_band: if =1 plot the 1std band together with data
   plot_residuals: if =1 plot residuals after convergence
   plot_distribtion: if =1 plot the bootstrapped distributions of the parameters
   save_figs: if =1 save the figures in png insted of displaying them
   show_progressbar: if =1 show the progress bar

   return the optimal vales of the parameters, their errors, 
   the value of chi^2,the number of dof, the p-value
   and the bootstrap samples of the parameters.
   """

   mask = ((x<=xmax) & (x>=xmin))
   x=x[mask]
   dx=dx[mask]
   y=y[mask]
   dy=dy[mask]

   band_size=1000

   data_length=len(x)

   true_param_length=len(params)

   extended_params=np.append(params, x)
 
   # array to store the bootstrapped results 
   boot_sample=np.empty((len(params), samples), dtype=np.float64)

   if plot_band==1:
     x_band=np.linspace(xmin, xmax, band_size)
     boot_band=np.empty((band_size, samples), dtype=np.float64)
  
   for i in range(samples):
     if show_progressbar==1: 
       pb.progress_bar(i, samples)

     # bootstrap sample
     bootx=x+np.random.normal(0, dx, data_length) 
     booty=y+np.random.normal(0, dy, data_length) 

     # least square regression
     ris = opt.leastsq(_residuals_xyerr, extended_params, ftol=stop_param, args=(bootx, dx, booty, dy, true_param_length, func))
     boot_sample[:,i]=ris[0][:true_param_length]
     if plot_band==1:
       boot_band[:,i]=func(x_band, ris[0][:true_param_length])

   # optimal parameters and errors
   ris=np.mean(boot_sample, axis=1)
   err=np.std(boot_sample, axis=1, ddof=1)

   # auxilliary stuff
   opt_res=_residuals_yerr(ris, x, y, dy, func)
   chi2=np.sum(opt_res*opt_res)
   dof=data_length - true_param_length
   pvalue=1.0 - stats.chi2.cdf(chi2, dof)

   if plot_fit==1:
     x_aux=np.linspace(xmin, xmax, 1000)
     y_aux=func(x_aux, ris)

     plt.figure('Best fit (chi2/dof=%.4f/%d=%f)' % (chi2, dof, chi2/dof))
     plt.xlim(0.9*xmin, 1.1*xmax)
     plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='ob', ms=5)
     plt.plot(x_aux,y_aux,'r-')

     if plot_band==1:
       band_mean=np.mean(boot_band, axis=1)
       band_std=np.std(boot_band, axis=1)
       plt.plot(x_band, band_mean + band_std,'g-')
       plt.plot(x_band, band_mean - band_std,'g-')

     if save_figs==1:
       plt.savefig('fit.png')
     else:
       plt.show()

   if plot_residuals==1:
     x_aux=np.linspace(xmin, xmax, 1000)
     y_aux=np.ones(len(x_aux))

     plt.figure('Residuals')
     plt.xlim(0.9*xmin, 1.1*xmax)
     plt.errorbar(x, opt_res, xerr=dx, yerr=1, fmt='ob', ms=5)
     plt.plot(x_aux, -2*y_aux, 'g:')
     plt.plot(x_aux, -y_aux, 'r--')
     plt.plot(x_aux, 0*y_aux, 'r-')
     plt.plot(x_aux, y_aux, 'r--')
     plt.plot(x_aux, 2*y_aux, 'g:')

     if save_figs==1:
       plt.savefig('residuals.png')
     else:
       plt.show()

   if plot_distribution==1:
     for i in range(0, len(params)):
       plt.figure('Bootstrapped distribution of param[%d]' % i)
       plt.xlabel('param[%d] values' % i)
       plt.ylabel('distribution histogram')
       plt.hist(boot_sample[i], bins='auto')

       if save_figs==1:
         plt.savefig('param'+str(i)+'.png')
       else:
         plt.show()

   return ris, err, chi2, dof, pvalue, boot_sample
 

def chi2_yerr(x, y, model, C_ij, *params):
    """
    Compute chi-squared for a model with correlated residuals.
    
    Parameters:
    - x: array-like, input values
    - y: array-like, observed values
    - model: callable, should accept (x, *params)
    - C_ij: 2D array, covariance matrix
    - *params: variable list of model parameters

    Returns:
    - chi2
    """
    residuals = y - model(x, *params)
    
    chi2 = np.sum(residuals * (np.linalg.inv(C_ij) @ residuals))

    return chi2


def format_uncertainty(value, dvalue, significance=2):
    """Creates a string of a value and its error in paranthesis notation, e.g., 13.02(45)"""
    if dvalue == 0.0 or (not np.isfinite(dvalue)):
        return str(value)
    if not isinstance(significance, int):
        raise TypeError("significance sstdneeds to be an integer.")
    if significance < 1:
        raise ValueError("significance needs to be larger than zero.")
    fexp = np.floor(np.log10(dvalue))
    if fexp < 0.0:
        return '{:{form}}({:1.0f})'.format(value, dvalue * 10 ** (-fexp + significance - 1), form='.' + str(-int(fexp) + significance - 1) + 'f')
    elif fexp == 0.0:
        return f"{value:.{significance - 1}f}({dvalue:1.{significance - 1}f})"
    else:
        return f"{value:.{max(0, int(significance - fexp - 1))}f}({dvalue:2.{max(0, int(significance - fexp - 1))}f})"

def format_uncertainty_vec(values, dvalues, significance=2):
    return [format_uncertainty(v, d, significance) for v, d in zip(values, dvalues)]


#***************************
# unit testing

if __name__=="__main__":
  
  print("**********************")
  print("UNIT TESTING")
  print()


  numsamples=1000

  length=10
  x=np.linspace(1, 10, length)
  y=2*x*x+np.random.normal(0, 10, length)
  dy=10*np.ones(length)

  print("Points generated with y=2*x*x + gaussian noise")
  print()

  fitparams=np.array([0.1, 1.2], dtype=np.float64)
  def parabfit(x, param):
    return param[0]+ x*x*param[1]

  print("Fit of the form param[0]+param[1]*x*x")
  print()

  print("Case with errors only on y")
  print()

  ris, err, chi2, dof, pvalue, boot_sample = fit_with_yerr(x, y, dy, 1, 10, parabfit, fitparams, numsamples)

  print("  chi^2/dof = {:3.3f}/{:d} = {:3.3f}".format(chi2, dof, chi2/dof))
  print("  p-value = %f" % pvalue)
  print()
  print("  a = {: f} +- {:f}".format(ris[0], err[0]) ) 
  print("  b = {: f} +- {:f}".format(ris[1], err[1]) ) 
  print()

  print("Case with errors both on x and y")
  print()

  dx=2.0*np.ones(length)/(length)

  ris, err, chi2, dof, pvalue, boot_sample = fit_with_xyerr(x, dx, y, dy, 1, 10, parabfit, fitparams, numsamples)

  print("  chi^2/dof = {:3.3f}/{:d} = {:3.3f}".format(chi2, dof, chi2/dof))
  print("  p-value = %f" % pvalue)
  print()
  print("  a = {: f} +- {:f}".format(ris[0], err[0]) ) 
  print("  b = {: f} +- {:f}".format(ris[1], err[1]) ) 
  print()
  print("**********************")

