import numpy as np
import matplotlib.pyplot as plt
import unyt as u
import h5py
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from scipy.interpolate import griddata
import os
import corner 
#own modules
from baoflamingo.fitting import BAO_fitter
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
        self.cfg=cfg
        self.filename = filename
        self.load_data(filename)

        #multipole creation
        if cfg.plotting.monopole or cfg.plotting.quadrupole:
            if cfg.plotting.monopole and cfg.plotting.quadrupole:
                l_list=[0,2]
            elif cfg.plotting.monopole and not cfg.plotting.quadrupole:
                l_list=[0]
            elif cfg.plotting.quadrupole and not cfg.plotting.monopole:
                l_list=[2]
            pole_proj = multipole_projector(
                xi=self.xi_data,
                cov=self.cov_data,
                mu=self.mu_data,
                s=self.s_data,
                ell_list=l_list, #0=monopole,2=quadrupole
                mu_range_minus1_to1=True, #can change in the future if needed but not right now
                #regularize=cfg.plotting.covariance_regularization
            )
            if cfg.plotting.monopole:
                self.mono_data,self.mono_data_err=pole_proj.multipoles[0],pole_proj.errors[0]
            if cfg.plotting.quadrupole:
                self.quad_data,self.quad_data_err=pole_proj.multipoles[2],pole_proj.errors[2]
     

        #rebinning if needed
        #subject to be introduced in the cfg file, for now set to 1 (no rebinning)
        self._rebin_all(s_rebin=s_rebin, mu_rebin=mu_rebin)
        
        #bao settings
        BAO_window = cfg.plotting.bao_window*u.Mpc
        #window around the expected BAO position for plotting, determines the 
        #colormesh scale, important and configured in yaml file
        self.mask_BAO = (self.s_data > self.BAO - 0.5*BAO_window) & (self.s_data < self.BAO + 0.5*BAO_window)
        
        
        #plot the bao ridge spline if wanted in the 2d correlation plot
        self.bao_ridge = cfg.plotting.plot_bao


        # --- Setting up the fitting class ---
        self.fit=BAO_fitter(s_template=self.s_template.value,
                 mono_template=self.mono_template, mono_template_err=None,
                 quad_template=self.quad_template, quad_template_err=None,)

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
            #data loading
            self.s_data = load_dataset("s_bin_centers")
            self.mu_data = load_dataset("mu_bin_centers")
            self.xi_data = load_dataset("xi")
            self.cov_data = load_dataset("cov") if "cov" in f else np.ones_like(self.xi_data)
            self.BAO = load_dataset("BAO_distance")
            self.H_z= load_dataset("effective_H_z")
            self.D_a= load_dataset("effective_D_a")
            #print(f"H_z={self.H_z}, D_a={self.D_a}, BAO position={self.BAO}")


            #template loading
            self.s_template=load_dataset("template_s")
            self.mono_template=load_dataset("template_xi0")
            self.quad_template=load_dataset("template_xi2")

            #loading other stuff
            self.name=load_dataset("name")

            #loading the parameters to save in the output file set in the config
            for param in self.cfg.fiducial.parameters_to_save:
                setattr(self, param, load_dataset(param))
            #loading the parameters for MCMC grid if manual cosmology is enabled 
            if self.cfg.fiducial.manual_cosmo:
                n_parameters=len(self.cfg.fiducial.parameters_mcmc)
                if n_parameters==1:
                    self.para_name=self.cfg.fiducial.parameters_mcmc[0]
                    setattr(self, 'para_value', load_dataset(self.para_name))
                elif n_parameters==2:
                    self.para1_name=self.cfg.fiducial.parameters_mcmc[0]
                    self.para2_name=self.cfg.fiducial.parameters_mcmc[1]
                    setattr(self, 'para1_value', load_dataset(self.para1_name))
                    setattr(self, 'para2_value', load_dataset(self.para2_name))

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
        self.auto_covar = np.diag(self.cov_data)
        self.auto_covar = self.rebin(array=self.auto_covar, factor=mu_rebin, axis=1)
        self.auto_covar = self.rebin(array=self.auto_covar, factor=s_rebin, axis=0)
        
        #2D correlation
        self.xi_data = self.rebin(array=self.xi_data, factor=mu_rebin, axis=1)
        self.xi_data = self.rebin(array=self.xi_data, factor=s_rebin, axis=0)
        
        #multipoles
        self.mono_data =self.rebin(array=self.mono_data,factor=s_rebin,axis=0)
        self.mono_data_err =self.rebin(array=self.mono_data_err,factor=s_rebin,axis=0)

        self.quad_data =self.rebin(array=self.quad_data,factor=s_rebin,axis=0)
        self.quad_data_err =self.rebin(array=self.quad_data_err,factor=s_rebin,axis=0)
        
        #mu and s
        self.mu_data = self.rebin(self.mu_data, mu_rebin, axis=0)
        self.s_data = self.rebin(self.s_data, s_rebin, axis=0)


    def _plot_2d_correlation(self):
        """Plot 2D ξ(s, μ) heatmap with optional BAO ridge spline."""
        

        plt.figure(figsize=(8,6))
        vlim = np.nanmax(np.abs(self.xi_data[self.mask_BAO, :]))
        im = plt.pcolormesh(self.s_data, self.mu_data, self.xi_data.T, shading="auto", cmap="seismic",
                            vmin=-vlim, vmax=vlim)
        plt.colorbar(im, label="ξ(s, μ)")
        plt.xlabel(f"s [{self.s_data.units}] (Comoving)")
        plt.ylabel("μ")
        plt.title("2D correlation function")

        # BAO ridge
        if self.bao_ridge:
            
            s_bao_max = np.array([self.s_data[self.mask_BAO][np.nanargmax(self.xi_data[self.mask_BAO, i])] for i in range(len(self.mu_data))])

            # weights from ls_std
            var_ridge = np.nanmean(self.ls_std[self.mask_BAO, :]**2, axis=0)
            weights = 1.0 / var_ridge

            spline = UnivariateSpline(self.mu_data, s_bao_max, w=weights, s=2000)
            mu_smooth = np.linspace(self.mu_data.min(), self.mu_data.max(), 500)
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
        im = plt.pcolormesh(self.s_data, self.mu_data, self.auto_covar.T,
             shading="auto", cmap="viridis",
             vmin=np.nanmin(self.auto_covar[self.mask_BAO, :]),
             vmax=np.nanmax(self.auto_covar[self.mask_BAO, :]))
        plt.colorbar(im, label="Var[ξ(s, μ)]")
        plt.xlabel(f"s [{self.s_data.units}]")
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

        s_data = self.s_data[self.mask_BAO] 
        mono_data = self.mono_data[self.mask_BAO]
        mono_data_err=self.mono_data_err[self.mask_BAO] 
        
        
        plt.figure(figsize=(8,6))
        
        #plot expected bao position with a vline
        plt.axvline(self.BAO.value, color='orange',
         linestyle='--', label=f'Expected BAO Position={self.BAO.value:.1f} {self.BAO.units}')
        
        #errorbar plot of the data
        plt.errorbar(s_data, mono_data,
        yerr=mono_data_err,color='green', label='Data',marker='x',linestyle='None',capsize=3)
        
        #gaussian fit plot
        #gaussian fitting for the 1d correlation of the bao signal
        if self.cfg.plotting.fit_gaussian:
            gaussian_results = self.fit.gaussian(
                                        s_data=s_data.value, 
                                        mono_data=mono_data,
                                        mono_data_err=mono_data_err, 
                                        init_amplitude=0.005, 
                                        init_mean=self.BAO.value, 
                                        init_stddev=5
                                        )
            if gaussian_results["fit"] is not None:
                gauss_fit=gaussian_results["fit"]
                gauss_mu=(gaussian_results["mu"],gaussian_results["mu_err"])
                gauss_sigma=(gaussian_results["sigma"],gaussian_results["sigma_err"])

                plt.scatter(s_data, gauss_fit, color='red', 
                label=f'Gaussian Fit, μ={gauss_mu[0]:.2f} +/- {gauss_mu[1]:.2f} Mpc, σ={gauss_sigma[0]:.2f} +/- {gauss_sigma[1]:.2f}Mpc')
            
            else:
                print("No fit for the gaussian.")
            
        
        
        #no shift
        if self.cfg.plotting.fit_noshift:
            # ---template ----
            template_no_shift_results=self.fit.template_no_shift(
                s_data=s_data.value, 
                mono_data=mono_data, 
                mono_data_err=mono_data_err,
                include_nuissance=self.cfg.plotting.include_nuissance, 
                poly_order=self.cfg.plotting.nuissance_poly_order
                )

            if template_no_shift_results["fit"] is not None:  
                fit_no_shift=template_no_shift_results["fit"]
                chi2_no_shift=template_no_shift_results["chi2"]
                
                plt.plot(s_data,fit_no_shift,
                        color="red",
                        label=f"template with NO shift,chi:{chi2_no_shift:.2f}")

            else:
                print("No fit for the template with no shift.")

        if self.cfg.plotting.fit_shift:
            #with shift  
            template_with_shift_results=self.fit.template_with_shift(
                s_data=s_data.value, 
                mono_data=mono_data, 
                mono_data_err=mono_data_err,
                include_nuissance=self.cfg.plotting.include_nuissance, 
                poly_order=self.cfg.plotting.nuissance_poly_order
            )
            
            if template_with_shift_results["fit"] is not None:
                fit_with_shift=template_with_shift_results["fit"]
                chi2_with_shift=template_with_shift_results["chi2"]
                self.alpha_with_shift=(template_with_shift_results["alpha"],
                                template_with_shift_results["alpha_err"])
            
            
                plt.plot(s_data,fit_with_shift,
                        color="orange",
            label=f"template with shift,alpha:{self.alpha_with_shift[0]:.3f}")
            
            else:
                print("No fit for the template with shift.")



        plt.xlabel(f"s [{self.s_data.units}]")
        plt.ylabel("ξ_0(s)")
        plt.title("Monopole ξ_0")
        plt.legend()
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_mono.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        plt.close()

        print(f"Monopole plot saved to {filename_plot}")

    
    def _plot_quadrupole(self):
        """monopole of the correlation"""

        s = self.s_data[self.mask_BAO] 
        quad_data = self.quad_data[self.mask_BAO]
        quad_data_err=self.quad_data_err[self.mask_BAO] 
        
        
        plt.figure(figsize=(8,6))
        #errorbar plot of the data
        plt.errorbar(s, quad_data,
        yerr=quad_data_err,color='green', label='Data',marker='x',linestyle='None',capsize=3)
        #template
        #plt.plot(self.s_template, self.quad_template,color="black",label="template")
        #rough mean and stddev of the quadrupole array in the bao region, simple std
        self.mu_quad,self.std_quad=np.average(quad_data,weights=1/quad_data_err**2),np.std(quad_data)
        
        plt.xlabel(f"s [{self.s_data.units}]")
        plt.ylabel("ξ_2(s)")
        plt.title("Quadrupole ξ_2")
        plt.legend()
        #plotting
        filename_plot=str(self.filename)
        filename_plot=filename_plot.replace('.hdf5','_quad.png')
        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300)
        plt.close()

        print(f"Quadrupole plot saved to {filename_plot}")



class posterior_plotter:

    def __init__(self, 
                cfg,
                redshift,
                mcmc_list, 
                outdir=".",
                use_quad_likelihood=False,
                true_pars=None,
                provided_likelihoods=None):
        """
        Parameters
        ----------
        mcmc_list : list of dicts
            Your grid entries containing α_mean, α_std, quad_mean, quad_std, and parameters.
        outdir : str
            Directory to save plots
        use_quad_likelihood : bool
            If True: include quadrupole likelihood contribution to posterior.
        true_pars : dict or None
            Example:
               {"para": 0.315}              # 1 parameter
               {"para1": 0.315, "para2": 0.685}   # 2 parameters
        """
        
        self.provided_likelihoods = provided_likelihoods
        if provided_likelihoods is None:
            self.redshift=redshift
            self.mcmc_list = mcmc_list
            self.outdir = outdir
            self.use_quad = use_quad_likelihood
            self.true_pars = true_pars
        else:
            self.redshift="combined"
            self.mcmc_list = mcmc_list
            self.outdir = outdir
            self.use_quad = use_quad_likelihood
            self.true_pars = true_pars

        #unpacking all the parameters in the configuration file
        self.ndim=len(cfg.fiducial.parameters_mcmc)

        if self.ndim == 1:
            self.p_vals=np.linspace(*cfg.fiducial.para_1_range, cfg.fiducial.points_per_para)
            self._plot_1d()
        elif self.ndim == 2:
            self.p1_vals = np.linspace(*cfg.fiducial.para_1_range, cfg.fiducial.points_per_para)
            self.p2_vals = np.linspace(*cfg.fiducial.para_2_range, cfg.fiducial.points_per_para)
            self._plot_2d()
        
        ("printed all the plots and posterior")

    # ------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------
    def _likelihood(self):
        """Gaussian likelihood from α (and ξ2 if enabled)."""
        """
        Returns a 2D array of likelihoods on a full param1 x param2 grid,
        with zero likelihood for missing points.
        """
        if self.ndim == 1:
            L_grid = np.zeros(len(self.p_vals))
        if self.ndim == 2:
            L_grid = np.zeros((len(self.p1_vals), len(self.p2_vals)))
        for d in self.mcmc_list:
            if self.ndim == 2:
                i = np.where(self.p1_vals == d["para1_value"])[0][0]
                j = np.where(self.p2_vals == d["para2_value"])[0][0]
            if self.ndim == 1:
                i = np.where(self.p_vals == d["para_value"])[0][0]
                
            
                
            L = np.exp(-0.5*((d["alpha_mean"]-1)/d["alpha_std"])**2)
            if self.use_quad:
                L *= np.exp(-0.5*((d["quad_mean"])/d["quad_std"])**2)
            if self.ndim == 2: 
                L_grid[i,j] = L
            if self.ndim == 1:
                L_grid[i] = L
           
        return L_grid

    # ------------------------------------------------------------
    # Compute posterior values for each grid point
    # ------------------------------------------------------------
    def _posterior(self):
        """Compute posterior from likelihood."""
        # normalize, might not be smart because of incorrect priors
        
        posterior = self.likelihood / np.sum(self.likelihood)
        return posterior  
    # ------------------------------------------------------------
    # Plotting: 1 PARAMETER
    # ------------------------------------------------------------
    def _plot_1d(self):
        self.para_value   = np.array([d["para_value"] for d in self.mcmc_list])
        self.para_name  = self.mcmc_list[0]["para_name"]
        
        if self.provided_likelihoods is None:
            self.alpha  = np.array([d["alpha_mean"] for d in self.mcmc_list])
            self.alpha_std  = np.array([d["alpha_std"]  for d in self.mcmc_list])
            self.quad   = np.array([d["quad_mean"] for d in self.mcmc_list])
            self.quad_std  = np.array([d["quad_std"]  for d in self.mcmc_list])
            

            # -----------------------
            # α plot
            # -----------------------
            plt.figure(figsize=(8,6))
            plt.errorbar(self.para_value, self.alpha, 
            yerr=self.alpha_std, fmt='o', capsize=3
            )
            
            if self.true_pars is not None:
                plt.axvline(self.true_pars["para"], color="black", ls="--", lw=0.5)

            plt.xlabel(self.para_name)
            plt.ylabel(r"$\alpha$")
            plt.title(f"Redshift {self.redshift:.2f} – α")
            
            plt.savefig(os.path.join(self.outdir, "a-alpha_mcmc.png"), dpi=300)
            plt.close()

            # -----------------------
            # Quadrupole plot
            # -----------------------
            plt.figure(figsize=(8,6))
            plt.errorbar(self.para_value,self.quad, 
            yerr=self.quad_std, fmt='o', capsize=3)

            if self.true_pars is not None:
                plt.axvline(self.true_pars["para"], color="black", ls="--", lw=0.5)

            plt.xlabel(self.para_name)
            plt.ylabel(r"Average Quadrupole $\xi_2$")
            plt.title(f"Redshift {self.redshift:.2f} – Quadrupole")
            
            plt.savefig(os.path.join(self.outdir, "a-quad_mcmc.png"), dpi=300)
            plt.close()

        # -----------------------
        # Posterior 1D
        # -----------------------
        if self.provided_likelihoods is None:
            self.likelihood = self._likelihood()

        #for use in the combined posterior
        else:
            self.likelihood = np.array(self.provided_likelihoods)
        posterior = self._posterior()
        sort = np.argsort(self.para)

        plt.figure(figsize=(8,6))
        plt.plot(self.para[sort], posterior[sort], "-o")
        if self.true_pars is not None:
            plt.axvline(self.true_pars["para"], color="black", ls="--", lw=0.5)

        plt.xlabel(self.para_name)
        plt.ylabel("Posterior")
        if self.provided_likelihoods is not None:
            plt.title(f"Redshift {self.redshift:.2f} – 1D Posterior")
        else:
            plt.title("Combined 1D Posterior")
        
        plt.savefig(os.path.join(self.outdir, "a-posterior_1d.png"), dpi=300)
        plt.close()



    # ------------------------------------------------------------
    # Plotting: 2 PARAMETERS
    # ------------------------------------------------------------
    def _plot_2d(self):
        
        self.para1_value  = np.array([d["para1_value"] for d in self.mcmc_list])
        self.para2_value  = np.array([d["para2_value"] for d in self.mcmc_list])
        self.para1_name = self.mcmc_list[0]["para1_name"]
        self.para2_name = self.mcmc_list[0]["para2_name"]
        
        if self.provided_likelihoods is None:
            self.alpha  = np.array([d["alpha_mean"] for d in self.mcmc_list])
            self.alpha_std  = np.array([d["alpha_std"] for d in self.mcmc_list])

            self.quad   = np.array([d["quad_mean"] for d in self.mcmc_list])
            self.quad_std   = np.array([d["quad_std"] for d in self.mcmc_list])        

            # ============================================================
            # α–1 scatter heatmap
            # ============================================================
            plt.figure(figsize=(8,6))
            sc = plt.scatter(self.para1_value, self.para2_value, 
                            c=self.alpha-1, cmap="seismic", 
                            s=120, vmin=-0.1, vmax=0.1)
            plt.colorbar(sc, label=r"$\alpha - 1$")

            if self.true_pars is not None:
                plt.axvline(self.true_pars["para1"], color="black", ls="--", lw=0.5)
                plt.axhline(self.true_pars["para2"], color="black", ls="--", lw=0.5)
                plt.scatter(self.true_pars["para1"], self.true_pars["para2"],
                            s=160, color="black", marker="x")

            plt.xlabel(self.para1_name)
            plt.ylabel(self.para2_name)
            plt.title(f"Redshift {self.redshift:.2f} – α Contour")
            
            plt.savefig(os.path.join(self.outdir, "a-alpha_contour.png"), dpi=300)
            plt.close()

            # ============================================================
            # Quadrupole scatter heatmap
            # ============================================================
            plt.figure(figsize=(8,6))
            sc = plt.scatter(self.para1_value, self.para2_value, 
                            c=self.quad, cmap="Reds", 
                            s=120, 
                            vmin=np.min(self.quad),
                            vmax=np.mean(self.quad))
            plt.colorbar(sc, label=r"Quadrupole $\xi_2$")

            if self.true_pars is not None:
                plt.axvline(self.true_pars["para1"], color="black", ls="--", lw=0.5)
                plt.axhline(self.true_pars["para2"], color="black", ls="--", lw=0.5)
                plt.scatter(self.true_pars["para1"], self.true_pars["para2"],
                            s=160, color="black", marker="x")

            plt.xlabel(self.para1_name)
            plt.ylabel(self.para2_name)
            plt.title(f"Redshift {self.redshift:.2f} – Quadrupole")
            
            plt.savefig(os.path.join(self.outdir, "a-quad_contour.png"), dpi=300)
            plt.close()

        # ============================================================
        # Posterior 2D contour
        # ============================================================
        if self.provided_likelihoods is None:
            self.likelihood = self._likelihood()
            
        #for use in the combined posterior
        else:
            self.likelihood = np.array(self.provided_likelihoods)
        posterior = self._posterior()

        #set up the grid for contourf
        P1, P2 = np.meshgrid(self.p1_vals, self.p2_vals, indexing="ij")
        # sanity check
        if posterior.shape != (len(self.p1_vals), len(self.p2_vals)):
            raise ValueError(
                f"Posterior has shape {posterior.shape}, but expected "
                f"({len(self.p1_vals)}, {len(self.p2_vals)})"
            )
        plt.figure(figsize=(8,6))
        cs = plt.contourf(
                    P1, P2, posterior,
                    levels=40,
                    cmap="viridis"
                    )
        plt.colorbar(cs, label="Posterior")

        if self.true_pars is not None:
            plt.scatter(self.true_pars["para1"], self.true_pars["para2"],
                        s=160, color="white", marker="x")

        plt.xlabel(self.para1_name)
        plt.ylabel(self.para2_name)
        if self.provided_likelihoods is not None:
            plt.title("Combined 2D Posterior")
        else:
            plt.title(f"Redshift {self.redshift:.2f} – Posterior 2D")
        
        plt.savefig(os.path.join(self.outdir, "a-posterior_2d_smoothed.png"), dpi=300)
        plt.close()

        # ============================================================
        # Corner plot

        
        
        samples = np.column_stack([
            P1.ravel(),
            P2.ravel()
        ])

        corner_posterior = posterior.ravel()
        #hardcoded, purely visual smoothing of the corner plot
        smooth=0.7
        


        corner_kwargs = dict(
            data=samples,
            weights=corner_posterior,
            labels=[self.para1_name, self.para2_name],
            levels=[0.68, 0.95],
            plot_contours=True,
            fill_contours=True,
            color="C1",
            show_titles=True,
            smooth=smooth,
            bins=[
                len(np.unique(self.para1_value)),
                len(np.unique(self.para2_value))
            ],
            hist_kwargs={"density": True, "align": "mid"}
        )

        # Only add truth lines if provided
        if self.true_pars is not None:
            corner_kwargs.update(
                truths=[self.true_pars["para1"], self.true_pars["para2"]],
                truth_color="k"
            )

        fig = corner.corner(**corner_kwargs)

        fig.savefig(os.path.join(self.outdir,"a-corner_posterior.png"), dpi=300)

