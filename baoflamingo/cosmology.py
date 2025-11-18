import numpy as np
import scipy.spatial as ss
from scipy.integrate import quad
from astropy.constants import c
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
import unyt as u
import astropy.units as au



class cosmo_tools:
    def __init__(self,
    box_size,
    constants,
    redshift,
    redshift_bin_width):
        #for colossus cosmology class
        if constants.name is not None:
            self.name=constants.name
        else: #for the swiftsimio cosmology class
            self.name=type(constants).__name__
        self.box_size=box_size
        #Hubble constant

        #strip units if necessary, always in km/s/Mpc for fiducial and real cosmology
        if isinstance(constants.H0, au.quantity.Quantity):
            self.H0 = constants.H0.value
        else:
            self.H0 = constants.H0

  
        self.h = self.H0 / 100.0
        #total matter
        self.Omega_m = constants.Om0
        self.Omh2 = self.Omega_m * self.h**2
        #baryons
        self.Omega_b = constants.Ob0
        self.Ombh2 = self.Omega_b * self.h**2
        #dark energy
        self.Omega_lambda = constants.Ode0
        #cmb
        if isinstance(constants.Tcmb0, au.quantity.Quantity):
            self.Tcmb = constants.Tcmb0.value
        else:
            self.Tcmb = constants.Tcmb0
        #neutrinos
        self.Neff = constants.Neff
       
        #radiation
        self.Omega_gamma=2.472e-5 * (self.Tcmb / 2.7255)**4 /(self.h)**2
        self.Omega_r = self.Omega_gamma * (1.0 + 0.2271 * self.Neff)
        #curvature
        
        self.Omega_k = 1.0 - self.Omega_m - self.Omega_r - self.Omega_lambda
        
        #constants
        self.c_km_s = c.to('km/s').value


        #save all the importan parameters here
        self.redshift=redshift
        
        self.bao_distance=self._bao_sound_horizon()
    
        
        

        #set outer and inner edges of the redshift bin
        self.bin_width=redshift_bin_width
        self._edges_bin()

        #---function to transform D_c to z ---
        #call cosmology.cosmo_tools.comoving_distance_to_redshift(redshift)
        self.comoving_distance_to_redshift=self._comoving_distance_to_redshift()

        #complete sphere and the maximum angle, handy to store here:
        self._observer_position()


        



    
        #internal functions for distances, do not change z here!!
    def E(self, z):
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        
        return np.sqrt(
            self.Omega_m * (1 + z)**3 +
            self.Omega_r * (1 + z)**4 +
            self.Omega_lambda +
            self.Omega_k * (1 + z)**2
        )

    @staticmethod
    def redshift_with_error(z):
        #randomly distributes point in the z (radial) axis
        #we ignore systematic errors for now
        sigma_z=0.0005*(1+z)
        random_z=np.random.normal(z,sigma_z)
        return random_z

    
        
    def _edges_bin(self):
        half_binwidth=self.bin_width/2
        self.min_redshift=self.redshift-half_binwidth
        self.max_redshift=self.redshift+half_binwidth
        self.outer_edge_bin=self.comoving_distance(self.max_redshift)
        self.inner_edge_bin=self.comoving_distance(self.min_redshift)
        self.center_bin=self.comoving_distance(self.redshift)
        self.delta_dr=self.outer_edge_bin-self.inner_edge_bin

    def _observer_position(self):
        if self.outer_edge_bin < 0.5 * self.box_size:
            self.complete_sphere=True
            self.max_angle=np.pi
        else:
            self.complete_sphere=False
            self.max_angle=np.arcsin(self.box_size / (2 *self.outer_edge_bin))*u.rad

    def comoving_distance(self, z):
        """
        Compute comoving line-of-sight distance D_C(z) in Mpc.
        Works for scalar or array z.
        """
        z=np.atleast_1d(z)
        Dc_list = []

        for zi in z:
            integral, _ = quad(lambda zp: 1.0 / self.E(zp), 0.0, zi, epsrel=1e-6)
            Dc_i = (self.c_km_s / self.H0) * integral
            Dc_list.append(Dc_i)

        Dc_array = np.array(Dc_list) * u.Mpc
        return Dc_array if len(Dc_array) > 1 else Dc_array[0]


    def luminosity_distance(self,z):
        """
        Compute luminosity distance D_L(z) in Mpc.
        """
        Dc = self.comoving_distance(z)
        Dl = Dc * (1+z)
        return Dl
    def angular_diameter_distance(self,z):
        """
        Compute angular diameter distance D_A(z) in Mpc.
        """
        Dc = self.comoving_distance(z)
        Da = Dc / (1 + z)
        return Da

    def _bao_sound_horizon(self):
        """
        Compute BAO comoving sound horizon r_d at the drag epoch.
        Based on Eisenstein & Hu (1998).
        """
        # --- Drag epoch redshift ---
        b1 = 0.313 * self.Omh2**(-0.419) * (1 + 0.607 * self.Omh2**0.674)
        b2 = 0.238 * self.Omh2**0.223
        z_drag = 1291 * self.Omh2**0.251 / (1 + 0.659 * self.Omh2**0.828) *\
         (1 + b1 * self.Ombh2**b2)

        # --- Sound horizon integral ---
        def R_of_z(zp):
            return (3.0 * self.Omega_b) / (4.0 * self.Omega_gamma) / (1.0 + zp)

        def c_s(zp):  
            return self.c_km_s / np.sqrt(3.0 * (1.0 + R_of_z(zp)))

        def integrand(zp):
            return c_s(zp) / (self.H0 *self.E(zp))

        r_d, _ = quad(integrand, z_drag, 1e7, epsrel=1e-6, limit=200)
        return r_d*u.Mpc

    def _comoving_distance_to_redshift(self):
        #builds a mapping of z <--> D_c
        #maybe increase resolution??
        z_grid = np.linspace(self.min_redshift, self.max_redshift, int(1e6)) 
        Dc_grid = np.array([self.comoving_distance(z).value for z in z_grid])  # Mpc
        # ensure monotonic
        assert np.all(np.diff(Dc_grid) > 0)
        inv_interp = PchipInterpolator(Dc_grid, z_grid, extrapolate=False)
        # call inv_interp(Dc_array) -> z_array (or raises for out-of-range)
        return inv_interp 

    #calculating effective cosmological functions within a certain redshift bin
    @staticmethod
    def effective_redshift(z):
        return np.mean(z)

    def effective_angular_diameter_distance(self,z):
        return self.angular_diameter_distance(self.effective_redshift(z))
    
    def effective_comoving_distance(self,z):
        return self.comoving_distance(self.effective_redshift(z))
    
    def effective_hubble_constant(self,z):
        return self.H0 * self.E(self.effective_redshift(z))*u.Unit('km/s/Mpc')