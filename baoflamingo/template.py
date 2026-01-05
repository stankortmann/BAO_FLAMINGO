import camb
import numpy as np
from mcfit import P2xi
from numpy.polynomial.legendre import legval

class template_CAMB:
    def __init__(self, cosmo, effective_redshift,
                 s_array, ell_list=[0, 2], non_linear=False):
        """
        Generate BAO templates (monopole + quadrupole) from CAMB for a given cosmology.
        
        Parameters
        ----------
        cosmo : Astropy cosmology object
        s_array : np.ndarray
            Array of separations [Mpc] to evaluate xi(s)
        ell_list : tuple
            List of multipoles to compute (default (0,2))
        """
        #defining a new, detailed s_array to fit later on in the plotting class
        s_min,s_max,s_len=s_array[0],s_array[-1], len(s_array)
        s_diff=s_max-s_min
        self.s = np.linspace(s_min-0.1*s_diff,
                             s_max+0.1*s_diff,
                             s_len*20)
        
        self.z= effective_redshift
        self.ell_list = ell_list
        self.non_linear=non_linear

        #actual running
        self._unpack_cosmo(cosmo)
        self._set_camb_params()
        self._get_powerspectrum()
        self._compute_multipoles()

    def _unpack_cosmo(self, cosmo):
        """Automatically extract parameters from an Astropy cosmology made in
        the pipeline."""
        self.H0 = cosmo.H0
        self.h = self.H0/ 100
        self.Ombh2 = cosmo.Obh2
        self.Omch2 = cosmo.Och2
        #give some standard values if not mentioned in cosmo initialization
        self.Omk = getattr(cosmo, "Ok0", 0.0)
        self.Tcmb = getattr(cosmo, "Tcmb0", 2.73) 
        self.Neff = getattr(cosmo, "Neff", 3.046) 
        self.wa = getattr(cosmo, "wa", 0.0)
        self.w0 = getattr(cosmo, "w0", -1) 
        

    def _set_camb_params(self):
        """Initialize CAMB parameters and compute P(k)."""
        # --- simple setup 
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=self.H0, ombh2=self.Ombh2,
                                omch2=self.Omch2, omk=self.Omk,
                                TCMB=self.Tcmb, nnu=self.Neff)

        #pretty standard and well established parameters
        self.pars.InitPower.set_params(As=2e-9, ns=0.965)

        # --- Dark energy setup if dynamical---
        if self.w0 !=-1:
            
            self.pars.DarkEnergy = camb.dark_energy.DarkEnergyPPF()
            self.pars.DarkEnergy.set_params(w=self.w0, wa=self.wa)

        self.pars.set_matter_power(redshifts=[self.z], kmax=10.0)
    
        self.pars.WantTransfer = True
        self.pars.WantCls = False
        self.pars.WantDerivedParameters = True
        self.results = camb.get_results(self.pars)
    def _get_powerspectrum(self):
        # Get linear matter power spectrum
        kh, z, pk = self.results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints=500)
        self.kh = kh      # h/Mpc
        self.pk = pk[0]   # at z=z_eff

    def _compute_multipoles(self):
        """Compute monopole and quadrupole templates using FFTLog."""
        self.xi_ell = {}
        self.xi_err = {}  # placeholder if needed later

        for ell in self.ell_list:
            P2xi_obj = P2xi(k=self.kh,l=ell, lowring=False)
            r,xi = P2xi_obj(self.pk, extrap=True)
            # Interpolate to requested s array, multiply by h because s is expected in Mpc/h
            xi_interp = np.interp(self.s*self.h, r, xi)
            self.xi_ell[ell] = xi_interp

    def get_multipoles(self):
        """Return a dict of multipoles xi_ell(s)."""
        return self.xi_ell

