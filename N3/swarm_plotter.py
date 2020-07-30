####
# generic plotter for two swarm plots
# have to configure npy file names and indices of data to plot
####
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


def make_swarms(
    d,
    save_filename=None,
    show=True,
    palette=None,  # for example, sns.xkcd_palette(["swamp green", "vermillion"]),
    ylim=None,
    xticks=None,
    ylabel="",
):
    """
    d: a list where each item in the list is a list of values corresponding to on 'swarm'
    """
    num_plots = len(d)
    # fig = plt.figure(figsize=[3*num_plots, 3.5])
    ax = plt.gca()

    print("len(d): ", len(d), " ", [len(i) for i in d])
    dat = np.concatenate(d)

    # organize and plot mis
    data_inds = np.concatenate([[i] * len(d[i]) for i in range(num_plots)])
    data = np.vstack([data_inds, dat]).T
    data = pd.DataFrame(data=data, columns=["type", "info"])

    if palette is not None:
        palette = sns.xkcd_palette(palette)

    sns.swarmplot(
        "type", "info", data=data, order=np.arange(num_plots), ax=ax, palette=palette
    )

    plt.xlabel(None)
    plt.ylabel("")
    plt.xticks(np.arange(num_plots), xticks)
    plt.ylim(ylim)
    plt.ylabel(ylabel)

    ax = plt.gca()
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)

    if show:
        plt.show()
