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

    def __init__(self, 
                filename,
                cfg,
                mu_rebin=1,
                s_rebin=1
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

        #rebinning if needed
        #subject to be introduced in the cfg file, for now set to 1 (no rebinning)
        self._rebin_all(s_rebin=s_rebin, mu_rebin=mu_rebin)

        #bao settings
        bao = cfg.plotting.expected_bao_position*u.Mpc
        bao_window = cfg.plotting.bao_window*u.Mpc
        #window around the expected BAO position for plotting, determines the 
        #colormesh scale, important and configured in yaml file
        self.mask_bao = (self.s > bao - bao_window) & (self.s < bao + bao_window)
        #this is a wide window only for the 1d correlation plot, hardcoded though
        bao_window_1d = 80*u.Mpc
        self.mask_s_1d_bao = (self.s > bao - bao_window_1d) & (self.s < bao + bao_window_1d)
        #plot the bao ridge spline if wanted in the 2d correlation plot
        self.bao_ridge = cfg.plotting.plot_bao

        #plotting if you want to plot the following statistics
        if cfg.plotting.correlation_2d:
            self._plot_2d_correlation()
        if cfg.plotting.variance_2d:    
            self._plot_2d_variance()
        if cfg.plotting.correlation_1d:
            self._plot_s_1d_correlation()
            self._plot_mu_1d_correlation()


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

            #will have to be implemented later on when this is passed on in the .hdf5 file!!!!
            #self.bao_position = load_dataset("bao_position")
            

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

    def _rebin_all(self,s_rebin=1, mu_rebin=1):
        #variance rebinning
        self.var_xi = self.ls_std**2
        self.var_xi = self.rebin(array=self.var_xi, factor=mu_rebin, axis=1)
        self.var_xi = self.rebin(array=self.var_xi, factor=s_rebin, axis=0)

        #correlation rebinning
        self.xi = self.rebin(array=self.xi, factor=mu_rebin, axis=1)
        self.xi = self.rebin(array=self.xi, factor=s_rebin, axis=0)

        #mu and s rebinning
        self.mu = self.rebin(self.mu, mu_rebin, axis=0)
        self.s = self.rebin(self.s, s_rebin, axis=0)


    def _plot_2d_correlation(self):
        """Plot 2D ξ(s, μ) heatmap with optional BAO ridge spline."""
        

        plt.figure(figsize=(8,6))
        vlim = np.nanmax(np.abs(self.xi[self.mask_bao, :]))
        im = plt.pcolormesh(self.s, self.mu, self.xi.T, shading="auto", cmap="seismic",
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
        

        
        plt.figure(figsize=(8,6))
        im = plt.pcolormesh(self.s, self.mu, self.var_xi.T,
             shading="auto", cmap="viridis",
             vmin=np.nanmin(self.var_xi[self.mask_bao, :]),
             vmax=np.nanmax(self.var_xi[self.mask_bao, :]))
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

    def _plot_s_1d_correlation(self):
        """Average ξ(s, μ) over μ and plot 1D correlation."""

        xi_mean = np.average(self.xi,weights=1/self.var_xi, axis=1)
        s_plot = self.s[self.mask_s_1d_bao] 
        xi_plot = xi_mean[self.mask_s_1d_bao] 

        plt.figure(figsize=(8,6))
        plt.plot(s_plot, xi_plot)
        plt.xlabel(f"s [{self.s.units}]")
        plt.ylabel("Σμ ξ(s, μ)")
        plt.title("1D correlation of ξ(s, μ) --> ξ(s)")
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_s_1d_correlation.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        print(f"1D correlation plot saved to {filename_plot}")

    def _plot_mu_1d_correlation(self):
        """Average ξ(s, μ) over μ and plot 1D correlation."""

        xi_mean = np.average(self.xi[self.mask_s_1d_bao],weights=1/self.var_xi[self.mask_s_1d_bao],  axis=0)
        #might introduce a mask here as well, but for mu it's not that necessary at the moment
        mu_plot = self.mu 
        xi_plot = xi_mean

        plt.figure(figsize=(8,6))
        plt.plot(mu_plot, xi_plot)
        plt.xlabel("μ")
        plt.ylabel("Σs ξ(s, μ)")
        plt.title("1D correlation of ξ(s, μ) --> ξ(μ)")
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_mu_1d_correlation.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        print(f"1D correlation plot saved to {filename_plot}")
       

