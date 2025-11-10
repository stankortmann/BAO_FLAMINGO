import yaml
import numpy as np

# Our own modules
import baoflamingo.data_structure_pycorr as ds
from baoflamingo.pipeline_single_pycorr import run_pipeline_single
from baoflamingo.plotting_pycorr import plot_correlation_2d


############-------ACTUAL RUNNING, DO NOT DELETE!!! -------#############

cfg=None
data_filename = "/cosma/home/do012/dc-kort1/BAO/results_angular_3/L1000N3600/HYDRO_FIDUCIAL/single_slice_snapshot_63.hdf5"
data_plot = plot_correlation_2d(
    cfg=cfg,
    filename=data_filename,
    save_plot=True,
    mu_rebinning=4, #main parameter to change in this visualisation plot!!!
    s_rebinning=2
)


