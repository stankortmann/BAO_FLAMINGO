import numpy as np
import scipy.spatial as ss
from scipy.integrate import quad
from astropy.constants import c
from scipy.optimize import curve_fit
from numba import njit, prange



#--- NUMBA NJIT speed up for coordinate transformations


@njit(parallel=True)
def cartesian_to_spherical_numba(coords, shift, observer, box_size=1000.0, apply_periodic=True):
    """
    Convert Cartesian (x, y, z) positions to spherical (r, theta, phi)
    relative to a given observer, optionally applying periodic boundaries.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates.
    observer : np.ndarray, shape (3,)
        Observer position.
    box_size : float
        Box size for periodic boundaries.
    apply_periodic : bool
        Whether to apply periodic boundary conditions.

    Returns
    -------
    sph_coords : np.ndarray, shape (N, 3)
        Columns are (r, theta, phi)
    """
    N = coords.shape[0]
    sph_coords = np.empty((N, 3), dtype=np.float64)
    
    for i in prange(N):
        x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]

        

        # shift coordinates if the sphere is complete, if not nothing happens
        dx = x - shift[0] 
        dy = y - shift[1] 
        dz = z - shift[2]

        # Apply periodic boundary conditions if needed
        if apply_periodic:
            dx = dx % box_size
            dy = dy % box_size
            dz = dz % box_size

        # Shift to be relative to the observer
        dx = dx - observer[0] 
        dy = dy - observer[1] 
        dz = dz - observer[2]

        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        theta = np.arccos(dz / r)  # polar angle [0, pi]
        phi = np.arctan2(dy, dx)   # azimuthal angle [-pi, pi]

        sph_coords[i, 0] = r
        sph_coords[i, 1] = theta
        sph_coords[i, 2] = phi

    return sph_coords




class coordinate_tools:

    def __init__(self,coordinates,box_size=1000,
        complete_sphere=True,observer=None,shift=None):
        
        self.coordinates=coordinates
        self.box_size=box_size
        self.complete_sphere=complete_sphere
        self.centre=np.array([box_size/2,box_size/2,box_size/2])
        self.observer=observer
        self.shift=shift

        # Compute spherical coordinates immediately and store
        self.spherical = cartesian_to_spherical_numba(
            self.coordinates, self.shift, self.observer,
            box_size=self.box_size, apply_periodic=self.complete_sphere
        )

    def to_spherical(self):
        """
        calls the numba function to speed up
        """
        

        return cartesian_to_spherical_numba(self.coordinates,self.shift,
                                            self.observer,
                                            box_size=self.box_size,
                                            apply_periodic=self.complete_sphere)

    @staticmethod
    def theta_phi_to_unitvec(coords):
        """
        Convert spherical coordinates (theta, phi) to 3D unit vectors.
        theta: polar angle [0, pi]
        phi: azimuthal angle [0, 2pi]
        """
        theta, phi = coords[:,0], coords[:,1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack((x, y, z))

    @staticmethod
    def chord_to_angular_separation(chord_lengths):

        """
        Convert chord lengths on a unit sphere to angular separations in radians.
        
        Parameters
        ----------
        chord_lengths : array-like
            Chord lengths (0 to 2)
        
        Returns
        -------
        alpha : array-like
            Angular separations in radians (0 to pi)
        """
        chord_lengths = np.asarray(chord_lengths)
        #clip for numerical stability
        alpha_rad = 2.0 * np.arcsin(np.clip(chord_lengths / 2.0, 0.0, 1.0)) 
        alpha_degrees= np.rad2deg(alpha_rad)
        return alpha_degrees
    


class correlation_tools:
    def __init__(self, box_size=1000.0, radius=427.0, max_angle_plus_dr=10.0, 
                 min_distance=0 , max_distance=250, bao_distance=150.0,
                 complete_sphere=False, 
                 seed=12345, n_random=50000,
                 bins=100, distance_type='euclidean'):
        
       
        self.box_size = box_size
        self.complete_sphere = complete_sphere
        self.radius = radius
        #only if incomplete sphere we need an angle limit for the randoms
        self.max_angle_plus_dr=None if complete_sphere else max_angle_plus_dr

        #we are converting distances to chord distances on a unit sphere
        self.min_chord = min_distance / self.radius
        self.max_chord = max_distance / self.radius
        self.bao_chord = bao_distance / self.radius
        
        #either euclidean distances in Mpc or angular distances in degrees
        self.distance_type = distance_type



        # Initialize random number generator
        self.rng = np.random.default_rng(seed)

        # Generate random catalog
        self.n_random = n_random
        self.unit_random = self._generate_random_catalog()
        self.tree_random = ss.cKDTree(self.unit_random, boxsize=None)
        

        #angles if necesarry
        self.min_angle = coordinate_tools.chord_to_angular_separation(self.min_chord) 
        self.max_angle = coordinate_tools.chord_to_angular_separation(self.max_chord) 
        self.bao_angle = coordinate_tools.chord_to_angular_separation(self.bao_chord)

        self.rr_angle = coordinate_tools.chord_to_angular_separation(self.rr_chord) 

        #--- Histogram set-up -----
        self.bins = bins
        if self.distance_type=='euclidean':
            self.bin_array = np.linspace(min_distance, max_distance, bins + 1)
        if self.distance_type=='angles':
            self.bin_array = np.linspace(0, self.max_angle, bins + 1)
        
        #already do rr  and the bin edges
        self.rr_counts,self.bin_edges = np.histogram(self.rr, bins=self.bin_array)
        self.rr_normalized = self.rr_counts / (self.n_random * (self.n_random - 1) / 2)

        ##Bin centers for plotting
        #This is actually not accurate for non linear bins!!!!!!!!
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_width = np.diff(self.bin_edges)[0]  # width of each bin


        

        
        

    
    #Always the same random catalogue for consistency
    def _generate_random_catalog(self):
        """Generate random points in spherical coordinates."""
        if self.complete_sphere:
            max_cos_theta = 1.0
            max_phi = np.pi
        else:
            max_cos_theta = self.box_size / (2 * self.plus_dr)
            max_phi = np.arcsin(self.box_size / (2 * self.plus_dr))
        
        random_theta = np.arccos(self.rng.uniform(low=-max_cos_theta, 
                                                  high=max_cos_theta,
                                                  size=self.n_random))
        random_phi = self.rng.uniform(low=-max_phi, high=max_phi, size=self.n_random)
        random = np.column_stack((random_theta, random_phi))
        unit_random=coordinate_tools.theta_phi_to_unitvec(self.random)
        return unit_random
    
    
    
    def tree_creation(coords):
        """
        Create a cKDTree from given coordinates.
        """

        ### No more periodic boundaries on the unit sphere or imcomplete sphere
        tree = ss.cKDTree(coords, boxsize=None)
        return tree

    
    def chord_distances_kdtree(tree1,tree2=None):
                    
    """
    Compute pairwise distances in 3D with optional periodic boundaries.
    Uses cKDTree for memory efficiency.
    """
    


        if tree2==None:
            
            # compute all pairwise distances
            # use sparse distance matrix to avoid full N^2 array
            ##This is for rr and dd
            sparse = tree1.sparse_distance_matrix(tree1, max_distance=self.max_chord,
                                                    output_type="coo_matrix")
            mask_auto=sparse.row < sparse.col
            dists = sparse.data[mask_auto]

        #this is for dr   
        else:
            
            sparse = tree1.sparse_distance_matrix(tree2, max_distance=self.max_chord,
                                                       output_type="coo_matrix")
            dists = sparse.data

        return dists

    #Call this once to get rr distances
    def rr(self):
        rr_chord=self.chord_distances_kdtree(tree1=self.tree_random)
        if self.distance_type=='angular':
            return coordinate_tools.chord_to_angular_separation(rr_chord)
        if self.distance_type=='euclidean':
            return self.radius*rr_chord
    
    def dd(self,coordinates):
        d_tree=self.tree_creation(coords=coordinates)
        dd_chord=self.chord_distances_kdtree(tree1=d_tree)
        if self.distance_type=='angular':
            return coordinate_tools.chord_to_angular_separation(dd_chord)
        if self.distance_type=='euclidean':
            return self.radius*dd_chord
        
    
    def dr(self, coordinates):
        d_tree=self.tree_creation(coords=coordinates)
        dr_chord=self.chord_distances_kdtree(tree1=d_tree, tree2=self.tree_random)
        if self.distance_type=='angular':
            return coordinate_tools.chord_to_angular_separation(dr_chord)
        if self.distance_type=='euclidean':
            return self.radius*dr_chord
    
    
    #----- The actual Landy-Szalay estimator -----
    
    def landy_szalay(self,coordinates):
        """
        Compute the Landy-Szalay estimator for the two-point correlation function.

        Parameters
        ----------
        dd : array-like
            Pairwise distances between data points.
        dr : array-like
            Pairwise distances between data and random points.
        rr : array-like
            Pairwise distances between random points.
        nbins : int
            Number of bins for histogramming distances.

        Returns
        -------
        bin_centers : np.ndarray
            Centers of the distance bins.
        w : np.ndarray
            Estimated two-point correlation function values.
        """
        
        n_data = coordinates.shape[0]

        # Histogram the distances
        dd_counts, _ = np.histogram(self.dd(coordinates), bins=self.bin_array)
        dr_counts, _ = np.histogram(self.dr(coordinates), bins=self.bin_array)
        

        
        
        norm_dd = n_data * (n_data - 1) / 2
        norm_dr = n_data * self.n_random
        

        dd_normalized = dd_counts / norm_dd
        dr_normalized = dr_counts / norm_dr
        

        # Avoid division by zero in case rr_normalized has zeros
        rr_nonzero = np.where(self.rr_normalized == 0, 1e-10, self.rr_normalized)
        if rr_nonzero.any()==1e-10:
            print("Warning: Some bins in RR have zero counts, adjusted to avoid division by zero.")

        # Landy-Szalay estimator
        w = (dd_normalized - 2 * dr_normalized + self.rr_normalized) / rr_nonzero

        

        return w


class cosmo_tools:
    def __init__(self, H0, Omega_m, Omega_lambda, Tcmb, Neff, Omega_b):
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

        # --- Drag epoch redshift ---
        self.b1 = 0.313 * self.Omh2**(-0.419) * (1 + 0.607 * self.Omh2**0.674)
        self.b2 = 0.238 * self.Omh2**0.223
        self.z_drag = 1291 * self.Omh2**0.251 / (1 + 0.659 * self.Omh2**0.828) *\
         (1 + self.b1 * self.Ombh2**self.b2)

        self.bao_sound_horizon=_bao_sound_horizon()

        #constants
        self.c_km_s = c.to('km/s').value



    

    def E(self, z):
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        
        return np.sqrt(
            self.Omega_m * (1 + z)**3 +
            self.Omega_r * (1 + z)**4 +
            self.Omega_lambda +
            self.Omega_k * (1 + z)**2
        )

    

    def comoving_distance(self, z):
        """
        Compute comoving line-of-sight distance D_C(z) in Mpc.
        """
        
        integral, _ = quad(lambda zp: 1.0 / self.E(zp), 0.0, z, epsrel=1e-6)
        Dc = (self.c_km_s / self.H0) * integral
        return Dc
    def luminosity_distance(self, z):
        """
        Compute luminosity distance D_L(z) in Mpc.
        """
        Dc = self.comoving_distance(z)
        Dl = (1 + z) * Dc
        return Dl
    def angular_diameter_distance(self, z):
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

        # --- Sound horizon integral ---
        def R_of_z(zp):
            return (3.0 * self.Omega_b) / (4.0 * self.Omega_gamma) / (1.0 + zp)

        def c_s(zp):  
            return self.c_km_s / np.sqrt(3.0 * (1.0 + R_of_z(zp)))

        def integrand(zp):
            return c_s(zp) / (self.H0 *self.E(zp))

        r_d, _ = quad(integrand, self.z_drag, 1e7, epsrel=1e-6, limit=200)
        return r_d




