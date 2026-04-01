#!/usr/bin/env python3

import matplotlib.pyplot as plt

color_dict = {
    0: "#000000",  # black
    1: "#3D5ACA",  # blue
    2: "#A71D60",  # pink
    3: "#C78A00",  # yellow
    4: "#4D3B79",  # purple
    5: "#BD4900",  # orange
    6: "#0072B2",  # new blue
    7: "#009E73",  # green
    8: "#D55E00",  # burnt orange
    9: "#CC79A7",  # magenta
}

bright_color_dict = {
    0: "#000000",  # black
    1: "#648fff",  # bright blue
    2: "#dc267f",  # bright pink
    3: "#ffb000",  # bright yellow
    4: "#785ef0",  # bright purple
    5: "#fe6100",  # bright orange
    6: "#648fff",  # bright blue
    7: "#00c49a",  # bright teal-green
    8: "#ff7f0e",  # bright burnt orange
    9: "#e377c2",  # bright magenta
}

marker_dict = {
    0: "s",  # square
    1: "o",  # circle
    2: "D",  # diamond
    3: "v",  # triangle down
    4: "^",  # triangle up
    5: "p",  # pentagon
    6: "*",  # star
    7: "X",  # X
    8: "<",  # triangle left
    9: ">",  # triangle right
}

lines_dict = {
    0: "solid",
    1: "dashed",
    2: "dashdot",
    3: "dotted",
    4: "solid",
    5: "dashed",
    6: "dashdot",
    7: "dotted",
    8: "solid",
    9: "dashed",
}


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.labelsize": 24,
        "font.size": 24,
    }
)


def data_nomarker(palette_idx=1, **kwargs) -> dict:
    try:
        color = color_dict[palette_idx]
    except KeyError:
        color = "darkblue"

    try:
        marker = marker_dict[palette_idx]
    except KeyError:
        marker = "o"

    pars = dict()
    pars["linestyle"] = "none"
    pars["color"] = color
    pars["markersize"] = 12
    pars["elinewidth"] = 1.5
    pars["capsize"] = 3.5
    pars["markeredgecolor"] = color
    pars["markerfacecolor"] = "none"
    pars.update(kwargs)

    return pars


def data(palette_idx=1, **kwargs) -> dict:
    try:
        color = color_dict[palette_idx]
    except KeyError:
        color = "darkblue"

    try:
        marker = marker_dict[palette_idx]
    except KeyError:
        marker = "o"

    pars = dict()
    pars["linestyle"] = "none"
    pars["color"] = color
    pars["marker"] = marker
    pars["markersize"] = 14
    pars["elinewidth"] = 2
    pars["markeredgewidth"] = 2
    pars["capsize"] = 5
    pars["markeredgecolor"] = color
    pars["markerfacecolor"] = "none"
    pars.update(kwargs)

    return pars


def data_noerr(palette_idx=1, **kwargs) -> dict:
    try:
        color = color_dict[palette_idx]
    except KeyError:
        color = "darkblue"

    try:
        marker = marker_dict[palette_idx]
    except KeyError:
        marker = "o"

    pars = dict()
    pars["linestyle"] = "none"
    pars["color"] = color
    pars["marker"] = marker
    pars["markersize"] = 12
    pars["markeredgewidth"] = 1.75
    pars["markeredgecolor"] = color
    pars["markerfacecolor"] = "none"
    pars.update(kwargs)

    return pars


def fit(palette_idx=0, **kwargs):
    try:
        linestyle = lines_dict[palette_idx]
    except KeyError:
        linestyle = "dashed"

    try:
        color = bright_color_dict[palette_idx]
    except KeyError:
        color = "black"

    pars = dict()
    pars["linestyle"] = linestyle
    pars["color"] = color
    pars["linewidth"] = 1.2
    pars.update(kwargs)
    return pars


def results(palette_idx=-1, **kwargs):
    try:
        color = color_dict[palette_idx]
    except KeyError:
        color = "red"

    try:
        marker = marker_dict[palette_idx]
    except KeyError:
        marker = "s"

    pars = dict()
    pars["linestyle"] = "none"
    pars["color"] = color
    pars["marker"] = marker
    pars["markersize"] = 8
    pars["elinewidth"] = 1.5
    pars["capsize"] = 3.5
    pars["markeredgecolor"] = color
    pars["markerfacecolor"] = color
    pars.update(kwargs)

    return pars


def conf_band(palette_idx=1, **kwargs):
    try:
        color = bright_color_dict[palette_idx]
    except KeyError:
        color = "blue"

    pars = dict()
    pars["facecolor"] = color
    pars["zorder"] = 1
    pars["alpha"] = 0.3
    pars.update(kwargs)
    return pars


def setup_mpl():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 36,
            "axes.titlesize": 32,
            "axes.labelsize": 32,
            "axes.labelpad": 16,
            "xtick.labelsize": 32,
            "ytick.labelsize": 32,
            "legend.fontsize": 24,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            "axes.linewidth": 1.2,
        }
    )


def style_axes(
    ax,
    ylog=False,
    xlog=False,
    xlabel=None,
    ylabel=None,
    title=None,
    legend=False,
    xlim=None,
    ylim=None,
    grid=False,
    tick_width=1.2,
    tick_length=6,
    minor_ticks=True,
    legend_kwargs=None,
):
    if ylog:
        ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log")

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(
        axis="both",
        which="major",
        width=tick_width,
        length=tick_length,
        pad=10,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=tick_width,
        length=0.3 * tick_length,
    )

    if minor_ticks:
        ax.minorticks_on()

    if grid:
        ax.grid(alpha=0.3)

    if legend:
        if legend_kwargs is None:
            legend_kwargs = {}
        ax.legend(**legend_kwargs)


if __name__ == "__main__":
    print("**********************")
    print("UNIT TESTING")
    print()

    import numpy as np
    from scipy.special import chebyt

    xxx = np.linspace(-1, 1, 10)
    rrr = 0.01 * np.ones(xxx.size)
    yy1 = chebyt(1)(xxx) + rrr * np.random.normal(size=xxx.size)
    yy2 = chebyt(2)(xxx) + rrr * np.random.normal(size=xxx.size)
    yy3 = chebyt(3)(xxx) + rrr * np.random.normal(size=xxx.size)
    yy4 = chebyt(4)(xxx) + rrr * np.random.normal(size=xxx.size)
    yy5 = chebyt(5)(xxx) + rrr * np.random.normal(size=xxx.size)

    x_plot = np.linspace(-1, 1, 500)
    y_plo1 = chebyt(1)(x_plot)
    y_plo2 = chebyt(2)(x_plot)
    y_plo3 = chebyt(3)(x_plot)
    y_plo4 = chebyt(4)(x_plot)
    y_plo5 = chebyt(5)(x_plot)

    x00 = 2 * np.pi
    r00 = 0.1
    y10 = 0
    y20 = 1

    plt.figure(0)
    plt.errorbar(xxx, yy1, rrr, **data(1), label="cheby1")
    plt.plot(x_plot, y_plo1, **fit(1))
    plt.errorbar(xxx, yy2, rrr, **data(2), label="cheby2")
    plt.plot(x_plot, y_plo2, **fit(2))
    plt.errorbar(xxx, yy3, rrr, **data(3), label="cheby3")
    plt.plot(x_plot, y_plo3, **fit(3))
    plt.errorbar(xxx, yy4, rrr, **data(4), label="cheby4")
    plt.plot(x_plot, y_plo4, **fit(4))
    plt.errorbar(xxx, yy5, rrr, **data(5), label="cheby5")
    plt.plot(x_plot, y_plo5, **fit(5))

    plt.fill_between(x_plot, y_plo1 - 0.1, y_plo1 + 0.1, **conf_band())

    plt.legend()

    plt.show()

    print("**********************")
