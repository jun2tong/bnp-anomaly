"""
======================================================================
Variational with birth and merge proposals for DP mixtures of Gaussians
======================================================================

How to train a DP mixture model.

We'll show that despite diverse, poor quality initializations, our proposal moves that insert new clusters (birth) and remove redundant clusters (merge) can consistently recover the same ideal posterior with 8 clusters.

"""
import bnpy
import numpy as np
import os
import sys
import pdb

from matplotlib import pylab
import seaborn as sns

# sphinx_gallery_thumbnail_number = 2

FIG_SIZE = (3, 3)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
# Generate time series with regime switching.
num_regime = 3
regime_noise = [0.1, 5, 1]
regime_loc = [10,5,0]
num_ts_samp = 5000  # number of data point
prop = [0.5, 0.3, 0.2]  # propotion of time in a regime

regime_prop = np.random.uniform(size=num_ts_samp)
obs = np.ones(num_ts_samp)

cum_prob = np.cumsum(prop)
for ii in range(num_regime):
    if ii > 0:
        mask_1 = regime_prop <= cum_prob[ii]
        mask_2 = regime_prop > cum_prob[ii-1]
        mask = mask_1 * mask_2
    else:
        mask = regime_prop <= cum_prob[ii]

    noise = np.random.normal(0, regime_noise[ii], np.sum(mask)) + regime_loc[ii]
    obs[mask] += noise
# obs[1000] = 10000
# pylab.plot(obs)

dataset =bnpy.data.XData(obs.reshape(-1,1))
###############################################################################
#
# Make a simple plot of the raw data

# pylab.plot(dataset.X[:, 0], dataset.X[:, 1], 'k.')
# pylab.gca().set_xlim([-2, 2])
# pylab.gca().set_ylim([-2, 2])
# pylab.tight_layout()


###############################################################################
#
# Setup: Function for visualization
# ---------------------------------
# Here's a short function to show the learned clusters over time.

def show_clusters_over_time(
        task_output_path=None,
        query_laps=[0, 1, 2, 5, 10, None],
        nrows=2):
    """ Read model snapshots from provided folder and make visualizations

    Post Condition
    --------------
    New matplotlib plot with some nice pictures.
    """
    ncols = int(np.ceil(len(query_laps) // float(nrows)))
    _, ax_handle_list = pylab.subplots(figsize=(FIG_SIZE[0] * ncols, FIG_SIZE[1] * nrows),
                                       nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for plot_id, lap_val in enumerate(query_laps):
        cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
        # Plot the current model
        cur_ax_handle = ax_handle_list.flatten()[plot_id]
        bnpy.viz.PlotComps.plotCompsFromHModel(
            cur_model, Data=dataset, ax_handle=cur_ax_handle)
        cur_ax_handle.set_xticks([-2, -1, 0, 1, 2])
        cur_ax_handle.set_yticks([-2, -1, 0, 1, 2])
        cur_ax_handle.set_xlabel("lap: %d" % lap_val)
    pylab.tight_layout()
    pylab.savefig('test.png')


###############################################################################
# Training from K=1 cluster
# -------------------------
# 
# Using 1 initial cluster, with birth and merge proposal moves.
K1_trained_model, K1_info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Gauss', 'memoVB',
    output_path='/tmp/AsteriskK8/trymoves-K=1/',
    nLap=100, nTask=1, nBatch=5,
    sF=0.1, ECovMat='eye',
    K=1, initname='randexamples',
    moves='birth,merge,shuffle',
    m_startLap=5, b_startLap=2, b_Kfresh=4)
pdb.set_trace()

# show_clusters_over_time(K1_info_dict['task_output_path'])
