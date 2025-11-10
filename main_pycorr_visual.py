import yaml
import numpy as np
from pathlib import Path

# Our own modules
import baoflamingo.data_structure_pycorr as ds
from baoflamingo.pipeline_single_pycorr import run_pipeline_single
import baoflamingo.plotting_pycorr_new as plt_pycorr


############-------ACTUAL RUNNING, DO NOT DELETE!!! -------#############

def make_path(results_dir, simulation, snapshot_num,
              base="/cosma/home/do012/dc-kort1/BAO"):
    """
    Create the full path for a snapshot file.

    Parameters
    ----------
    results_dir : str
        Directory like 'results_angular_4'
    simulation : str
        Simulation name like 'L1000N0900'
    snapshot_num : int or str
        Snapshot number, e.g. 63
    base : str
        Path to the BAO base directory
   

    Returns
    -------
    Path
        A pathlib.Path object pointing to the constructed file
    """
    snapshot_num = str(snapshot_num)
    filename = f"single_slice_snapshot_{snapshot_num}.hdf5"

    return Path(base) / results_dir / simulation / filename


cfg=None
data_filename = make_path(
    results_dir="results_angular_4",
    simulation="L1000N3600/HYDRO_FIDUCIAL",
    snapshot_num=65,
    base="/cosma/home/do012/dc-kort1/BAO"
)
data_plot = plt_pycorr.correlation_plotter(
    filename=data_filename,
    mu_rebin=1,
    s_rebin=1,
    bao=150,bao_window=20, #in comoving Mpc  
    plot_bao_ridge=True,
    plot_2d_correlation=True,
    plot_2d_variance=True,
    plot_1d_projection=True
    )


