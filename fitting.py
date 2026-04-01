import numpy as np
from scipy.optimize import curve_fit


def boot_fit(x, y, d_y, b_y, model, lim_inf, lim_sup, extension=None):
    x_fit, y_fit, d_y_fit, b_y_fit = (
        x[lim_inf:lim_sup],
        y[lim_inf:lim_sup],
        d_y[lim_inf:lim_sup],
        b_y[lim_inf:lim_sup],
    )
    opt, cov = curve_fit(model, x_fit, y_fit, sigma=d_y_fit, absolute_sigma=True)

    n_boot = len(b_y[0])

    x_linsp = np.linspace(x_fit[0], x_fit[-1], 51)

    if extension:
        x_linsp = np.linspace(extension[0], extension[1], extension[2])

    y_linsp = model(x_linsp, *opt)

    b_opt = []
    b_y_linsp = []

    for j in range(n_boot):
        y_fit_j = [b_y_fit[i][j] for i in range(len(b_y_fit))]

        opt_j, cov_j = curve_fit(
            model, x_fit, y_fit_j, sigma=d_y_fit, absolute_sigma=True
        )
        b_opt.append(opt_j)

        y_linsp_j = model(x_linsp, *opt_j)
        b_y_linsp.append(y_linsp_j)
    d_opt = np.std(b_opt, axis=0, ddof=1)

    c2r = np.sum(((model(x_fit, *opt) - y_fit) / d_y_fit) ** 2) / (
        len(x_fit) - len(opt)
    )

    d_y_linsp = np.std(b_y_linsp, axis=0, ddof=1)
    return x_linsp, y_linsp, d_y_linsp, opt, d_opt, b_opt, c2r
