import numpy as np
import matplotlib.pyplot as plt
import unyt as u
import h5py

def plot_correlation_2d(cfg, filename, save_plot=True):
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
    
    s_bao = 150
    window = 20  # Mpc around BAO scale

    # find indices in the BAO region
    mask_bao = (s.value > s_bao - window) & (s.value < s_bao + window)
    xi_bao_region = xi[mask_bao, :]

    max_abs = np.nanmax(np.abs(xi_bao_region))
    vmin, vmax = -max_abs, max_abs

    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    # transpose xi to align axes
    #vmin, vmax = np.nanpercentile(xi, [2, 98])  # e.g. 2nd–98th percentile range
    #vmin, vmax = -0.1,0.1 #overwritten, has to be adjusted on a per snapshot basis
    #needs more focus!!

    im = plt.pcolormesh(s, mu, xi.T, shading="auto", cmap="viridis",
                    vmin=vmin, vmax=vmax) 
    plt.colorbar(im, label="ξ(s, μ)")
    plt.xlabel(f"s [{s.units}]")
    plt.ylabel(f"μ")
    plt.title(filename.split("/")[-1])
    plt.grid(False)

    if save_plot:
        png_filename = filename.replace(".hdf5", "_2d_plot.png")
        plt.tight_layout()
        plt.savefig(png_filename, dpi=300)
        print(f"2D plot saved as '{png_filename}'")

    plt.show()

    return {"s_bin_centers": s, "mu_bin_centers": mu, "ls_avg": xi}
    
