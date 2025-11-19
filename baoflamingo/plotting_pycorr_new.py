import numpy as np
import matplotlib.pyplot as plt
import unyt as u
import h5py
from scipy.interpolate import UnivariateSpline

from baoflamingo.fitting import gaussian_data

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
        BAO_window = cfg.plotting.bao_window*u.Mpc
        #window around the expected BAO position for plotting, determines the 
        #colormesh scale, important and configured in yaml file
        self.mask_BAO = (self.s > self.BAO - 0.5*BAO_window) & (self.s < self.BAO + 0.5*BAO_window)
        #this is a wide window only for the 1d correlation plot, hardcoded though
        BAO_window_1d = 80*u.Mpc
        self.mask_s_1d_BAO = (self.s > self.BAO - 0.5*BAO_window_1d) & (self.s < self.BAO + 0.5*BAO_window_1d)
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
            self._plot_s_bins_1d_correlation(bins_per_plot=4)


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
            self.BAO = load_dataset("BAO_distance")
            self.H_z= load_dataset("effective_H_z")
            self.D_a= load_dataset("effective_D_a")
            print(f"H_z={self.H_z}, D_a={self.D_a}, BAO position={self.BAO}")

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
        vlim = np.nanmax(np.abs(self.xi[self.mask_BAO, :]))
        im = plt.pcolormesh(self.s, self.mu, self.xi.T, shading="auto", cmap="seismic",
                            vmin=-vlim, vmax=vlim)
        plt.colorbar(im, label="ξ(s, μ)")
        plt.xlabel(f"s [{self.s.units}] (Comoving)")
        plt.ylabel("μ")
        plt.title("2D correlation function")

        # BAO ridge
        if self.bao_ridge:
            
            s_bao_max = np.array([self.s[self.mask_BAO][np.nanargmax(self.xi[self.mask_BAO, i])] for i in range(len(self.mu))])

            # weights from ls_std
            var_ridge = np.nanmean(self.ls_std[self.mask_BAO, :]**2, axis=0)
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
        plt.close()
        print(f"2D correlation plot saved to {filename_plot}")
        

    def _plot_2d_variance(self):
        """Plot 2D variance (ls_std^2)."""
        

        
        plt.figure(figsize=(8,6))
        im = plt.pcolormesh(self.s, self.mu, self.var_xi.T,
             shading="auto", cmap="viridis",
             vmin=np.nanmin(self.var_xi[self.mask_BAO, :]),
             vmax=np.nanmax(self.var_xi[self.mask_BAO, :]))
        plt.colorbar(im, label="Var[ξ(s, μ)]")
        plt.xlabel(f"s [{self.s.units}]")
        plt.ylabel("μ")
        plt.title("Variance of ξ(s, μ)")
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_2d_covariance.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        plt.close()
        print(f"2D covariance plot saved to {filename_plot}")

    def _plot_s_1d_correlation(self):
        """Average ξ(s, μ) over μ and plot 1D correlation."""

        xi_mean = np.average(self.xi,weights=1/self.var_xi, axis=1)
        std_mean=np.sqrt(1/np.sum(1/self.var_xi, axis=1))
        s_plot = self.s[self.mask_BAO] 
        xi_plot = xi_mean[self.mask_BAO]
        std_plot=std_mean[self.mask_BAO] 
        
        #gaussian fitting for the 1d correlation of the bao signal
        gauss_fit,mu,sigma = gaussian_data(  distances=s_plot.value, 
                                    correlation=xi_plot,
                                    yerror=std_plot, 
                                    initial_amplitude=0.005, 
                                    initial_mean=self.BAO.value, 
                                    initial_stddev=5)
        
        plt.figure(figsize=(8,6))
        #plot expected bao position with a vline
        plt.axvline(self.BAO.value, color='orange',
         linestyle='--', label=f'Expected BAO Position={self.BAO.value:.1f} {self.BAO.units}')
        #errorbar plot of the data
        plt.errorbar(s_plot, xi_plot,
        yerr=std_plot,color='green', label='Data',marker='x',linestyle='None',capsize=3)
        #gaussian fit plot
        plt.scatter(s_plot, gauss_fit, color='red', 
        label=f'Gaussian Fit, μ={mu[0]:.2f} +/- {mu[1]:.2f} Mpc, σ={sigma[0]:.2f} +/- {sigma[1]:.2f}Mpc')
        
        plt.xlabel(f"s [{self.s.units}]")
        plt.ylabel("Σμ ξ(s, μ)")
        plt.title("1D correlation of ξ(s, μ) --> ξ(s)")
        plt.legend()
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_s_1d_correlation.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        plt.close()

        print(f"1D correlation plot saved to {filename_plot}")

    def _plot_mu_1d_correlation(self):
        """Average ξ(s, μ) over μ and plot 1D correlation."""

        # we will average over s in the BAO window with the correlation values as weights
        window_xi = np.maximum(self.xi[self.mask_BAO],0)#clipped so negative values do not interfere
        window_s=self.s[self.mask_BAO]
        window_s_2d = np.tile(window_s[:, None], (1, window_xi.shape[1]))  # shape (Ns_window, Nmu_window)
       

        # Compute delta s for each mu in the window around BAO
        average_s=np.average(window_s_2d, axis=0, weights=window_xi)
        delta_s_mu = self.BAO-average_s

      

        #xi_mean = np.average(self.xi[self.mask_s_1d_BAO],weights=1/self.var_xi[self.mask_s_1d_BAO],  axis=0)
        #might introduce a mask here as well, but for mu it's not that necessary at the moment
        mu_plot = self.mu 
        #xi_plot = xi_mean
        print(f"The average of delta_s over mu is {np.mean(delta_s_mu)}")
        positive_mu= (mu_plot >=0)
        plt.figure(figsize=(8,6))
        plt.scatter(mu_plot[positive_mu], delta_s_mu[positive_mu])
        plt.xlabel("μ")
        plt.ylabel("s-s_bao [Mpc]")
        plt.title("1D correlation of ξ(s, μ) --> ξ(μ)")
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_mu_1d_correlation.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        plt.close()
        print(f"1D correlation plot saved to {filename_plot}")
       

    def _plot_s_bins_1d_correlation(self, bins_per_plot=2):
        """
        Compute 1D correlation ξ(s) for subsets of positive μ bins and plot them.

        bins_per_plot = 2 → 4 plots total (2 μ-bins each)
        bins_per_plot = 4 → 2 plots total (4 μ-bins each)
        """

        # --- Select positive μ bins ---
        positive_mu_mask = (self.mu >= 0)
        mu_positive = self.mu[positive_mu_mask]       # shape = (Nmu,)
        xi_positive = self.xi[:, positive_mu_mask]    # shape = (Ns, Nmu)
        var_positive = self.var_xi[:, positive_mu_mask]

        nbins = len(mu_positive)
        

        # split into groups
        # Example for bins_per_plot = 2:
        # split indices [0,1], [2,3], [4,5], [6,7]
        indices = np.arange(nbins)
        groups = [indices[i:i+bins_per_plot] for i in range(0, nbins, bins_per_plot)]

        # Loop over μ-bin groups
        for gi, group in enumerate(groups):

            # --- Reduce ξ(s,μ) over the selected μ-bins ---
            xi_sub = xi_positive[:, group]
            var_sub = var_positive[:, group]

            # Weighted mean over μ
            xi_mean = np.average(xi_sub, axis=1, weights=1/var_sub)
            std_mean = np.sqrt(1 / np.sum(1/var_sub, axis=1))

            # Apply only s-mask (BAO window)
            s_plot = self.s[self.mask_BAO]
            xi_plot = xi_mean[self.mask_BAO]
            std_plot = std_mean[self.mask_BAO]

            # --- Gaussian fit with your existing function ---
            gauss_fit, mu_fit, sigma_fit = gaussian_data(
                distances=s_plot.value,
                correlation=xi_plot,
                yerror=std_plot,
                initial_amplitude=0.005,
                initial_mean=self.BAO.value,
                initial_stddev=5
            )

            # --- Plot ---
            plt.figure(figsize=(8,6))
            plt.axvline(self.BAO.value, color='orange', linestyle='--',
                        label=f'Expected BAO Position={self.BAO.value:.1f} {self.BAO.units}')

            plt.errorbar(
                s_plot, xi_plot, yerr=std_plot,
                color='green', marker='x', linestyle='None',
                capsize=3, label='Data'
            )

            plt.scatter(
                s_plot, gauss_fit, color='red',
                label=(f'Gaussian Fit, μ={mu_fit[0]:.2f} ± {mu_fit[1]:.2f} Mpc, '
                    f'σ={sigma_fit[0]:.2f} ± {sigma_fit[1]:.2f} Mpc')
            )

            # Group label
            mu_range_label = f"μ in [{mu_positive[group[0]]:.2f}, {mu_positive[group[-1]]:.2f}]"

            plt.xlabel(f"s [{self.s.units}]")
            plt.ylabel("Σμ ξ(s, μ)")
            plt.title(f"1D correlation for {mu_range_label}")
            plt.legend()

            # Save
            filename_plot = str(self.filename).replace(
                ".hdf5",
                f"_s_1d_correlation_mu_group{gi}.png"
            )
            plt.tight_layout()
            plt.savefig(filename_plot, dpi=300)
            plt.close()

            print(f"Saved μ-group {gi}: {filename_plot}")

