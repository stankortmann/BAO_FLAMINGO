import numpy as np
import matplotlib.pyplot as plt
import unyt as u
import h5py
from scipy.interpolate import UnivariateSpline

def plot_correlation_2d(cfg, filename, save_plot=True,mu_rebinning=1, s_rebinning=1):
    """
    Load a single-slice Landy-Szalay HDF5 file and plot the 2D correlation function ξ(s, μ)
    as a color-coded heatmap.

    ```
    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    save_plot : bool
        Whether to save the figure as PNG.

    Returns
    -------
    dict
        Dictionary containing loaded data:
        {"s_bin_centers": ..., "mu_bin_centers": ..., "ls_avg": ...}
    """

    # --- Load data ---
    with h5py.File(filename, "r") as f:
        data = {}
        for key in f.keys():
            dset = f[key]
            if "units" in dset.attrs:
                data[key] = dset[()] * u.Unit(dset.attrs["units"])
            else:
                data[key] = dset[()]

    s = data["s_bin_centers"]
    mu = data["mu_bin_centers"]
    xi = data["ls_avg"]  # shape: (len(s), len(mu))
    std= data["ls_std"] # shape: (len(s), len(mu))
    
    s_bao = 150
    window = 20 # Mpc around BAO scale

    # find indices in the BAO region
    mask_bao = (s.value > s_bao - window) & (s.value < s_bao + window)
    xi_bao_region = xi[mask_bao, :]

    max_abs = np.nanmax(np.abs(xi_bao_region))
    vmin, vmax = -max_abs, max_abs

    
    
    # reshape xi along mu dimension and average every two bins

    #first for mu
    xi_rebin = xi.reshape(xi.shape[0], -1, mu_rebinning).mean(axis=2)
    mu_rebin = mu.reshape(-1, mu_rebinning).mean(axis=1)

    #ow for s
    xi_rebin = xi_rebin.reshape(-1, s_rebinning, xi_rebin.shape[1]).mean(axis=1)
    s_rebin = s.reshape(-1, s_rebinning).mean(axis=1)

   




    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    

    im = plt.pcolormesh(s_rebin, mu_rebin, xi_rebin.T, shading="auto", cmap="seismic",
                    vmin=vmin, vmax=vmax) 
    plt.colorbar(im, label="ξ(s, μ)")
    plt.xlabel(f"s [{s.units}] (Comoving)")
    plt.ylabel(f"μ")
    plt.title(filename.split("/")[-1])
    # s-coordinate of max for each μ, plotting the BAO ridge
    s_bao_max = np.array([s[mask_bao][np.nanargmax(xi[mask_bao, i])] for i in range(len(mu))])

    # --- Spline fit ---
    spline = UnivariateSpline(mu, s_bao_max, s=2000)
    mu_smooth = np.linspace(mu.min(), mu.max(), 500)
    s_smooth = spline(mu_smooth)
    
    # Plot spline
    plt.plot(s_smooth, mu_smooth, color='green', 
    linestyle='--', linewidth=2, label="BAO ridge spline")
    plt.legend()
    
    #plt.plot(s_bao_max, mu, color='green', linestyle='--', linewidth=4, label="BAO ridge")
    plt.grid(False)

    if save_plot:

        #2d plot with rebinning saved here
        png_filename = filename.replace(".hdf5", "_2d_plot.png")
        plt.tight_layout()
        plt.savefig(png_filename, dpi=300)
        print(f"2D plot saved as '{png_filename}'")

        #1d plot summed over all mu with no rebinning
        plt.figure(figsize=(8, 6))
        png2_filename = filename.replace(".hdf5", "_1d_plot.png")
        plt.plot(s[mask_bao],np.sum(xi,axis=1)[mask_bao])
        plt.xlabel(f"s [{s.units}] (Comoving)")
        plt.savefig(png2_filename, dpi=300)
        print(f"1D plot saved as '{png2_filename}'")

    plt.show()

    return {"s_bin_centers": s, "mu_bin_centers": mu, "ls_avg": xi}
    
