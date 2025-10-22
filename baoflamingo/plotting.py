import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import unyt as u
import h5py

# own modules
import baoflamingo.statistics as stat
import baoflamingo.smooth_fitting as sf

def plot_correlation_single_slice(cfg,filename,show_fit=False, save_plot=True):
    """
    Load a single-slice Landy-Szalay HDF5 file and plot the correlation function.
    
    Parameters
    ----------
    filename_histogram : str
        Path to the HDF5 file.
    distance_min : unyt_quantity
        Minimum separation to plot.
    distance_max : unyt_quantity
        Maximum separation to plot.
    show_fit : bool
        Whether to overlay a smooth fit.
    save_plot : bool
        Whether to save the figure as PNG.
    
    Returns
    -------
    dict
        Dictionary containing loaded data and BAO peak info:
        {"bin_centers": ..., "ls_avg": ..., "ls_std": ..., "bao": ..., "r_bao": ..., "xi_bao": ...}
    """
    
    # --- Load data ---
    data = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            dset = f[key]
            if "units" in dset.attrs:
                data[key] = dset[()] * u.Unit(dset.attrs["units"])
            else:
                data[key] = dset[()]
    
    # Map datasets to local variables
    bin_centers = data["bin_centers"]
    bao = data["bao"]
    ls_avg = data["ls_avg"]
    ls_std = data["ls_std"]
    
    
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    plt.scatter(bin_centers, ls_avg, label="Landy–Szalay")
    
    # Smooth baseline
    spline = UnivariateSpline(bin_centers.value, ls_avg, s=20)
    baseline = spline(bin_centers.value)
    
    # Residual and BAO peak
    xi_residual = ls_avg - baseline
    peak_idx = np.argmax(xi_residual)
    r_bao = bin_centers[peak_idx]
    xi_bao = xi_residual[peak_idx]
    
    plt.plot(bin_centers, baseline, color='r', linestyle='--', label='Smooth baseline')
    plt.axvline(bao, color="g", linestyle="--", label=f"BAO angle = {bao:.2f}")
    
    print(f"BAO peak at r ~ {r_bao:.2f} with ξ ~ {xi_bao:.4f}")
    print(f"Expected BAO angle: {bao}")
    
    # Optional error bars
    # plt.errorbar(bin_centers, ls_avg, yerr=ls_std, fmt="x", alpha=0.7)
    
    # Optional smooth fit
    if show_fit:
        w_power, w_poly = sf.fit_smooth_correlation(bin_centers, ls_avg)
        plt.plot(bin_centers, w_power, label="Power-law fit", color="black")
        plt.plot(bin_centers, ls_avg - w_power, label="Data - Smooth", color="orange")
    
    # Labels & grid
    plt.xlabel(f"r [{bin_centers.units}]")
    plt.ylabel("ξ(r)")
    plt.title(filename.split("/")[-1])
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_plot:
        png_filename = filename.replace(".hdf5", "_plot.png")
        plt.tight_layout()
        plt.savefig(png_filename, dpi=300)
        print(f"Plot saved as '{png_filename}'")
    
    plt.show()
    
    # Return data dictionary
    return {
        "bin_centers": bin_centers,
        "ls_avg": ls_avg,
        "ls_std": ls_std,
        "bao": bao,
        "r_bao": r_bao,
        "xi_bao": xi_bao
    }
