import numpy as np
import scipy.spatial as ss
from scipy.integrate import quad
from astropy.constants import c
from scipy.optimize import curve_fit



class cosmo_tools:
    def __init__(self,box_size, H0, Omega_m, Omega_lambda, Tcmb, Neff, Omega_b, redshift,n_sigma):
        self.box_size=box_size
        #Hubble constant
        self.H0 = H0
        self.h = self.H0 / 100.0
        #total matter
        self.Omega_m = Omega_m
        self.Omh2 = self.Omega_m * self.h**2
        #baryons
        self.Omega_b = Omega_b
        self.Ombh2 = self.Omega_b * self.h**2
        #dark energy
        self.Omega_lambda = Omega_lambda
        #cmb
        self.Tcmb = Tcmb
        self.Neff = Neff
        #radiation
        self.Omega_gamma=2.472e-5 * (self.Tcmb / 2.7255)**4 /(self.h)**2
        self.Omega_r = self.Omega_gamma * (1.0 + 0.2271 * self.Neff)
        #curvature
        self.Omega_k = 1.0 - self.Omega_m - self.Omega_r - self.Omega_lambda
        #constants
        self.c_km_s = c.to('km/s').value

        # --- Drag epoch redshift ---
        self.b1 = 0.313 * self.Omh2**(-0.419) * (1 + 0.607 * self.Omh2**0.674)
        self.b2 = 0.238 * self.Omh2**0.223
        self.z_drag = 1291 * self.Omh2**0.251 / (1 + 0.659 * self.Omh2**0.828) *\
         (1 + self.b1 * self.Ombh2**self.b2)

        #save all the importan parameters here
        self.redshift=redshift
        
        self.redshift_error=self._redshift_error(n_sigma)
        self.dz=2*self.redshift_error #for later analysis needed
        self.bao_distance=self._bao_sound_horizon()
        self.comoving_distance=self._comoving_distance(self.redshift)
        self.plus_dr=self._comoving_distance(self.redshift+self.redshift_error)
        self.minus_dr=self._comoving_distance(self.redshift-self.redshift_error)
        self.delta_dr=self.plus_dr-self.minus_dr
        
        self.luminosity_distance=self._luminosity_distance()
        self.angular_diameter_distance=self._angular_diameter_distance()

        



    
        #internal functions for distances, do not change z here!!
    def E(self, z):
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        
        return np.sqrt(
            self.Omega_m * (1 + z)**3 +
            self.Omega_r * (1 + z)**4 +
            self.Omega_lambda +
            self.Omega_k * (1 + z)**2
        )

    def _redshift_error(self,n):
        #we will assume a 3 sigma redshift bin
        #we ignore systematic errors for now
        sigma_z=0.0005*(1+self.redshift)
        return n*sigma_z

    def _comoving_distance(self,z):
        """
        Compute comoving line-of-sight distance D_C(z) in Mpc.
        """
        
        integral, _ = quad(lambda zp: 1.0 / self.E(zp), 0.0, z, epsrel=1e-6)
        Dc = (self.c_km_s / self.H0) * integral
        return Dc
    def _luminosity_distance(self):
        """
        Compute luminosity distance D_L(z) in Mpc.
        """
        Dc = self.comoving_distance
        Dl = (1 + self.redshift) * Dc
        return Dl
    def _angular_diameter_distance(self):
        """
        Compute angular diameter distance D_A(z) in Mpc.
        """
        Dc = self.comoving_distance
        Da = Dc / (1 + self.redshift)
        return Da

    def _bao_sound_horizon(self):
        """
        Compute BAO comoving sound horizon r_d at the drag epoch.
        Based on Eisenstein & Hu (1998).
        """

        # --- Sound horizon integral ---
        def R_of_z(zp):
            return (3.0 * self.Omega_b) / (4.0 * self.Omega_gamma) / (1.0 + zp)

        def c_s(zp):  
            return self.c_km_s / np.sqrt(3.0 * (1.0 + R_of_z(zp)))

        def integrand(zp):
            return c_s(zp) / (self.H0 *self.E(zp))

        r_d, _ = quad(integrand, self.z_drag, 1e7, epsrel=1e-6, limit=200)
        return r_d