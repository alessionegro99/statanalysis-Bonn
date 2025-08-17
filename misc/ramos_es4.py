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


us = [6.5489, 5.8673, 5.3013, 4.4901, 3.8643, 3.2029, 2.7359, 2.3900, 2.1257]
sigmas = [14.005, 11.464, 9.371, 7.139, 5.622, 4.354, 3.541, 2.991, 2.575]
d_sigmas = [0.175, 0.123, 0.079, 0.047, 0.028, 0.019, 0.014, 0.010, 0.009]

def sspar(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**4

x_plot_fit = np.linspace(us[0], us[-1], 101)

plt.figure(figsize=(18,12))
plt.errorbar(us, sigmas, d_sigmas, **plot.data(1))

opt, cov = curve_fit(sspar, us, sigmas, sigma=d_sigmas, absolute_sigma=True)
chi2 = reg.chi2_corr(us, sigmas, sspar, np.diag(np.array(d_sigmas)**2), *opt)/(len(us) - len(opt))

y_plot_fit = sspar(x_plot_fit, *opt)

plt.plot(x_plot_fit, y_plot_fit, **plot.fit(1), label=fr'$\chi^2_r$={chi2:.2f}')

plt.legend()

plt.xlabel(r'$u$')
plt.ylabel(r'$\sigma$($u$)')

plt.show()

g2_mu = 11.31
for i in range(0,4):
    print(g2_mu)
    g2_2mu = sspar(g2_mu, *opt)
    g2_mu = g2_2mu