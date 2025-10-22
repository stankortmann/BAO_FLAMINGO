import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import unyt as u
import h5py

# own modules
import statistics as stat
import smooth_fitting as sf

def plot_correlation_single_slice(cfg,show_fit=False, save_plot=True):
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
        {"bin_centers": ..., "ls_avg": ..., "ls_std": ..., "bao_angle": ..., "r_bao": ..., "xi_bao": ...}
    """
    
    # --- Load data ---
    data = {}
    with h5py.File(filename_histogram, "r") as f:
        for key in f.keys():
            dset = f[key]
            if "units" in dset.attrs:
                data[key] = dset[()] * u.Unit(dset.attrs["units"])
            else:
                data[key] = dset[()]
    
    # Map datasets to local variables
    bin_centers = data["bin_centers"]
    bao_angle = data["bao"]
    ls_avg = data["ls_avg"]
    ls_std = data["ls_std"]
    
    # Create mask based on distance range
    mask = (bin_centers >= distance_min) & (bin_centers <= distance_max)
    bin_centers_plot = bin_centers[mask]
    ls_avg_plot = ls_avg[mask]
    ls_std_plot = ls_std[mask]
    
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    plt.scatter(bin_centers_plot, ls_avg_plot, label="Landy–Szalay")
    
    # Smooth baseline
    spline = UnivariateSpline(bin_centers_plot.value, ls_avg_plot.value, s=20)
    baseline = spline(bin_centers_plot.value)
    
    # Residual and BAO peak
    xi_residual = ls_avg_plot.value - baseline
    peak_idx = np.argmax(xi_residual)
    r_bao = bin_centers_plot[peak_idx]
    xi_bao = xi_residual[peak_idx]
    
    plt.plot(bin_centers_plot, baseline, color='r', linestyle='--', label='Smooth baseline')
    plt.axvline(bao_angle, color="g", linestyle="--", label=f"BAO angle = {bao_angle:.2f}")
    
    print(f"BAO peak at r ~ {r_bao:.2f} with ξ ~ {xi_bao:.4f}")
    print(f"Expected BAO angle: {bao_angle}")
    
    # Optional error bars
    # plt.errorbar(bin_centers_plot, ls_avg_plot, yerr=ls_std_plot, fmt="x", alpha=0.7)
    
    # Optional smooth fit
    if show_fit:
        w_power, w_poly = sf.fit_smooth_correlation(bin_centers, ls_avg, theta_min=None, theta_max=80)
        plt.plot(bin_centers_plot, w_power[mask], label="Power-law fit", color="black")
        plt.plot(bin_centers_plot, ls_avg_plot.value - w_power[mask], label="Data - Smooth", color="orange")
    
    # Labels & grid
    plt.xlabel(f"r [{bin_centers_plot.units}]")
    plt.ylabel("ξ(r)")
    plt.title(filename_histogram.split("/")[-1])
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_plot:
        png_filename = filename_histogram.replace(".hdf5", "_plot.png")
        plt.tight_layout()
        plt.savefig(png_filename, dpi=300)
        print(f"Plot saved as '{png_filename}'")
    
    plt.show()
    
    # Return data dictionary
    return {
        "bin_centers": bin_centers_plot,
        "ls_avg": ls_avg_plot,
        "ls_std": ls_std_plot,
        "bao_angle": bao_angle,
        "r_bao": r_bao,
        "xi_bao": xi_bao
    }
