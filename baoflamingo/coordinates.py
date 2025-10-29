import numpy as np
from numba import njit, prange
import unyt as u




#--- NUMBA NJIT speed up for coordinate transformations


@njit(parallel=True)
def cartesian_to_spherical_numba(coords, shift, observer, box_size=1000.0):
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

        

        # shift coordinates randdomly and apply boundary conditions
        dx = (x - shift[0])%box_size 
        dy = (y - shift[1])%box_size  
        dz = (z - shift[2])%box_size 

       

        # Shift to be relative to the observer
        dx = observer[0]-dx
        dy = observer[1]-dy
        dz = observer[2]-dz

        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        theta = np.arccos(dz / r)  # polar angle [0, pi]
        phi = np.arctan2(dy, dx)   # azimuthal angle [-pi, pi]

        sph_coords[i, 0] = r
        sph_coords[i, 1] = theta
        sph_coords[i, 2] = phi

    return sph_coords




class coordinate_tools:

    def __init__(self,cosmology,observer=None,shift=None):
        
        self.box_size=cosmology.box_size.value
        self.complete_sphere=cosmology.complete_sphere
        self.observer=observer
        self.shift=shift

    def cartesian_to_spherical(self,coordinates):
        #this setup to call this inside the class, 
        #numba functions cannot be inside!
        sph_coords=cartesian_to_spherical_numba(
            coords=coordinates,
            shift=self.shift,
            observer=self.observer,
            box_size=self.box_size)

        return sph_coords
    
    @staticmethod
    def spherical_to_cartesian_positions(sph_coords, cosmo):
        """
        Convert spherical coords (z,theta,phi) and redshift array -> Cartesian positions (N,3)
        using cosmology object `cosmo_to_use` that provides comoving_distance(z).
        Assumes sph_coords = array (N,2) with (theta, phi) in radians (same as your coordinate_tools).
        """
        
        # cosmo must provide comoving_distance(z) returning astropy Quantity in Mpc
        chi = cosmo.comoving_distance(sph_coords[:,0]) #shape (N)
        

        # convert (theta,phi) to unit vectors
        unit_vec = self.theta_phi_to_unitvec(sph_coords[:,1:])# shape (N,3)
        

        # Cartesian coordinates in Mpc
        pos = chi*unit_vec  # shape (N,3)
        return pos


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
        return np.column_stack((x, y, z)) #dimensionless

    @staticmethod
    def chord_to_angular_degrees(chord_lengths):

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
        return alpha_degrees*u.deg

    @staticmethod
    def chord_to_angular_radians(chord_lengths):

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
        
        return alpha_rad*u.rad


    @staticmethod
    def angular_radians_to_chord(alpha_rad):
        """
        Convert angular separations in radians to chord lengths on a unit sphere.
        
        Parameters
        ----------
        alpha_rad : array-like
            Angular separations in radians (0 to pi)
        
        Returns
        -------
        chord_lengths : array-like
            Chord lengths (0 to 2)
        """
        alpha_rad = np.asarray(alpha_rad)
        chord_lengths = 2.0 * np.sin(alpha_rad / 2.0)
        return chord_lengths

        
    @staticmethod
    def theta_phi_to_ra_dec(coords):
        theta = coords[:,0]
        phi = coords[:,1]
        dec = np.pi/2 - theta
        ra = phi
        coords_out = np.column_stack((ra, dec))
        return coords_out * u.rad
    
