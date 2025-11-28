import numpy as np
import matplotlib.pyplot as plt
import unyt as u
import h5py
from scipy.interpolate import UnivariateSpline

from baoflamingo.fitting import gaussian_data
from baoflamingo.multipole import multipole_projector

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
            Path to the HDF5 file containing s_bin_centers, mu_bin_centers, xi, cov.
        """
        self.filename = filename
        self.load_data(filename)

        #multipole creation
        pole_proj = multipole_projector(
            mu=self.mu,
            s=self.s,
            ell_list=(0,2), #0=monopole,2=quadrupole
            mu_range_minus1_to1=True, #can change in the future if needed but not right now
            regularize=cfg.plotting.covariance_regularization
        )
        self.mono_xi,self.mono_err=pole_proj.multipoles[0],pole_proj.errors[0]
        self.quad_xi,self.quad_err=pole_proj.multipoles[2],pole_proj.errors[2]
     

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
        if cfg.plotting.autocovariance_2d:    
            self._plot_2d_autocovariance()
        if cfg.plotting.monopole:
            self._plot_monopole()
        if cfg.plotting.quadrupole:
            self._plot_quadrupole()



    def load_data(self, filename):
        """Load data from HDF5 file."""
        with h5py.File(filename, "r") as f:
            def load_dataset(key):
                dset = f[key]
                return dset[()] * u.Unit(dset.attrs["units"]) if "units" in dset.attrs else dset[()]
            
            self.s = load_dataset("s_bin_centers")
            self.mu = load_dataset("mu_bin_centers")
            self.xi = load_dataset("xi")
            self.cov = load_dataset("cov") if "cov" in f else np.ones_like(self.xi)
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
        #2D autocovariance
        self.auto_covar = np.diag(self.cov)
        self.auto_covar = self.rebin(array=self.auto_covar, factor=mu_rebin, axis=1)
        self.auto_covar = self.rebin(array=self.auto_covar, factor=s_rebin, axis=0)
        
        #2D correlation
        self.xi = self.rebin(array=self.xi, factor=mu_rebin, axis=1)
        self.xi = self.rebin(array=self.xi, factor=s_rebin, axis=0)
        
        #multipoles
        self.mono_xi =self.rebin(array=self.mono_xi,factor=s_rebin,axis=0)
        self.mono_xi_err =self.rebin(array=self.mono_xi_err,factor=s_rebin,axis=0)

        self.quad_xi =self.rebin(array=self.quad_xi,factor=s_rebin,axis=0)
        self.quad_xi_err =self.rebin(array=self.quad_xi_err,factor=s_rebin,axis=0)
        
        #mu and s
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
        

    def _plot_2d_autocovariance(self):
        """Plot 2D variance (ls_std^2)."""
        

        
        plt.figure(figsize=(8,6))
        im = plt.pcolormesh(self.s, self.mu, self.auto_covar.T,
             shading="auto", cmap="viridis",
             vmin=np.nanmin(self.auto_covar[self.mask_BAO, :]),
             vmax=np.nanmax(self.auto_covar[self.mask_BAO, :]))
        plt.colorbar(im, label="Var[ξ(s, μ)]")
        plt.xlabel(f"s [{self.s.units}]")
        plt.ylabel("μ")
        plt.title("Autocovariance of ξ(s, μ)")
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_2d_autocovariance.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        plt.close()
        print(f"2D autocovariance plot saved to {filename_plot}")

    def _plot_monopole(self):
        """monopole of the correlation"""

        s = self.s[self.mask_BAO] 
        mono_xi = self.mono_xi[self.mask_BAO]
        mono_xi_err=self.mono_xi_err[self.mask_BAO] 
        
        
        plt.figure(figsize=(8,6))
        
        #plot expected bao position with a vline
        plt.axvline(self.BAO.value, color='orange',
         linestyle='--', label=f'Expected BAO Position={self.BAO.value:.1f} {self.BAO.units}')
        
        #errorbar plot of the data
        plt.errorbar(s, mono_xi,
        yerr=mono_xi_err,color='green', label='Data',marker='x',linestyle='None',capsize=3)
        
        #gaussian fit plot
        #gaussian fitting for the 1d correlation of the bao signal
        gauss_fit,mu,sigma = gaussian_data(  distances=svalue, 
                                    correlation=xi,
                                    yerror=mono_xi_err, 
                                    initial_amplitude=0.005, 
                                    initial_mean=self.BAO.value, 
                                    initial_stddev=5)
        plt.scatter(s_plot, gauss_fit, color='red', 
        label=f'Gaussian Fit, μ={mu[0]:.2f} +/- {mu[1]:.2f} Mpc, σ={sigma[0]:.2f} +/- {sigma[1]:.2f}Mpc')
        

        
        plt.xlabel(f"s [{self.s.units}]")
        plt.ylabel("ξ_0(s)")
        plt.title("Monopole ξ_0")
        plt.legend()
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_monopole.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        plt.close()

        print(f"Monopole plot saved to {filename_plot}")

    
    def _plot_quadrupole(self):
        """monopole of the correlation"""

        s = self.s[self.mask_BAO] 
        quad_xi = self.quad_xi[self.mask_BAO]
        quad_xi_err=self.quad_xi_err[self.mask_BAO] 
        
        
        plt.figure(figsize=(8,6))
        #plot expected bao position with a vline
        plt.axvline(self.BAO.value, color='orange',
         linestyle='--', label=f'Expected BAO Position={self.BAO.value:.1f} {self.BAO.units}')
        #errorbar plot of the data
        plt.errorbar(s, quad_xi,
        yerr=quad_xi_err,color='green', label='Data',marker='x',linestyle='None',capsize=3)
        
        
        plt.xlabel(f"s [{self.s.units}]")
        plt.ylabel("ξ_2(s)")
        plt.title("Quadrupole ξ_2")
        plt.legend()
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_quadrupole.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        plt.close()

        print(f"Quadrupole plot saved to {filename_plot}")



