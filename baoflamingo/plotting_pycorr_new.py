import numpy as np
import matplotlib.pyplot as plt
import unyt as u
import h5py
from scipy.interpolate import UnivariateSpline

class correlation_plotter:
    """
    Class to handle 2D correlation ξ(s, μ) plotting, variance plotting, 
    1D projection, and BAO ridge spline fitting.
    """

    def __init__(self, filename,
    mu_rebin=1,
    s_rebin=1,
    bao=150,bao_window=20, #in comoving Mpc  
    plot_bao_ridge=True,
    plot_2d_correlation=True,
    plot_2d_variance=False,
    plot_1d_projection=True
    ):
        """
        Load correlation data from HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file containing s_bin_centers, mu_bin_centers, ls_avg, ls_std.
        """
        self.filename = filename
        self.load_data(filename)
        self.mask_bao = (self.s.value > bao - bao_window) & (self.s.value < bao + bao_window)
        self.bao_ridge = plot_bao_ridge
        if plot_2d_correlation:
            self._plot_2d_correlation(mu_rebin=mu_rebin, s_rebin=s_rebin)
        if plot_2d_variance:    
            self._plot_2d_variance()
        if plot_1d_projection:
            self._plot_1d_projection()


    def load_data(self, filename):
        """Load data from HDF5 file."""
        with h5py.File(filename, "r") as f:
            def load_dataset(key):
                dset = f[key]
                return dset[()] * u.Unit(dset.attrs["units"]) if "units" in dset.attrs else dset[()]
            
            self.s = load_dataset("s_bin_centers")
            self.mu = load_dataset("mu_bin_centers")
            self.xi = load_dataset("ls_avg")
            self.ls_std = load_dataset("ls_std") if "ls_std" in f else np.ones_like(self.xi)

    @staticmethod
    def rebin(array, factor, axis=0):
        """
        Rebin a numpy array along a given axis by averaging over 'factor' elements.

        Parameters
        ----------
        array : np.ndarray
        factor : int
        axis : int
        """
        if factor == 1:
            return array
        shape = list(array.shape)
        n = shape[axis] // factor
        shape[axis] = n
        new_shape = shape[:axis] + [n, factor] + shape[axis+1:]
        rebinned = array.reshape(new_shape).mean(axis=axis+1)
        return rebinned

    def _plot_2d_correlation(self, mu_rebin=1, s_rebin=1):
        """Plot 2D ξ(s, μ) heatmap with optional BAO ridge spline."""
        xi_rebin = self.rebin(self.xi, mu_rebin, axis=1)
        xi_rebin = self.rebin(xi_rebin, s_rebin, axis=0)
        mu_rebin = self.rebin(self.mu, mu_rebin, axis=0)
        s_rebin = self.rebin(self.s, s_rebin, axis=0)

        plt.figure(figsize=(8,6))
        vlim = np.nanmax(np.abs(xi_rebin[self.mask_bao, :]))
        im = plt.pcolormesh(s_rebin, mu_rebin, xi_rebin.T, shading="auto", cmap="seismic",
                            vmin=-vlim, vmax=vlim)
        plt.colorbar(im, label="ξ(s, μ)")
        plt.xlabel(f"s [{self.s.units}] (Comoving)")
        plt.ylabel("μ")
        plt.title("2D correlation function")

        # BAO ridge
        if self.bao_ridge:
            
            s_bao_max = np.array([self.s[self.mask_bao][np.nanargmax(self.xi[self.mask_bao, i])] for i in range(len(self.mu))])

            # weights from ls_std
            var_ridge = np.nanmean(self.ls_std[self.mask_bao, :]**2, axis=0)
            weights = 1.0 / var_ridge

            spline = UnivariateSpline(self.mu, s_bao_max, w=weights, s=2000)
            mu_smooth = np.linspace(self.mu.min(), self.mu.max(), 500)
            s_smooth = spline(mu_smooth)
            plt.plot(s_smooth, mu_smooth, color="green", linestyle="--", linewidth=2, label="BAO ridge spline")
            plt.legend()

        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_2d_correlation.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        print(f"2D correlation plot saved to {filename_plot}")
        

    def _plot_2d_variance(self):
        """Plot 2D variance (ls_std^2)."""
        var_xi = self.ls_std
        plt.figure(figsize=(8,6))
        print(self.s)
        im = plt.pcolormesh(self.s, self.mu, var_xi.T,
             shading="auto", cmap="viridis",
             vmin=np.nanmin(var_xi[self.mask_bao, :]),
             vmax=np.nanmax(var_xi[self.mask_bao, :]))
        plt.colorbar(im, label="Var[ξ(s, μ)]")
        plt.xlabel(f"s [{self.s.units}]")
        plt.ylabel("μ")
        plt.title("Variance of ξ(s, μ)")
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_2d_covariance.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        print(f"2D covariance plot saved to {filename_plot}")

    def _plot_1d_projection(self):
        """Sum ξ(s, μ) over μ and plot 1D projection."""
        xi_mean = np.mean(self.xi, axis=1)
        s_plot = self.s[self.mask_bao] 
        xi_plot = xi_mean[self.mask_bao] 

        plt.figure(figsize=(8,6))
        plt.plot(s_plot, xi_plot)
        plt.xlabel(f"s [{self.s.units}]")
        plt.ylabel("Σμ ξ(s, μ)")
        plt.title("1D projection of ξ(s, μ)")
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_1d_correlation.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        print(f"1D correlation plot saved to {filename_plot}")
       

