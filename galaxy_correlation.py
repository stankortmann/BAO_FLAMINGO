import numpy as np
import scipy.spatial as ss
from scipy.integrate import quad
from astropy.constants import c
from scipy.optimize import curve_fit
from numba import njit, prange
import treecorr




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
        
        

        # Compute spherical coordinates immediately and store
        self.spherical = cartesian_to_spherical_numba(
            coordinates, shift, observer,
            box_size=box_size, apply_periodic=complete_sphere
        )


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
                 leafsize=50, 
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

        #The leafsize of the tree
        self.leafsize=leafsize

        # Initialize random number generator
        self.rng = np.random.default_rng(seed)

        # Generate random catalog
        self.n_random = n_random
        self.unit_random = self._generate_random_catalog()
        self.tree_random = self.tree_creation(self.unit_random)
        

        #angles if necesarry
        self.min_angle = coordinate_tools.chord_to_angular_separation(self.min_chord) 
        self.max_angle = coordinate_tools.chord_to_angular_separation(self.max_chord) 
        self.bao_angle = coordinate_tools.chord_to_angular_separation(self.bao_chord)
        

        

        #--- Histogram set-up -----
        self.bins = bins
        if self.distance_type=='euclidean':
            self.bin_array = np.linspace(min_distance, max_distance, bins + 1)
        if self.distance_type=='angular':
            self.bin_array = np.linspace(0, self.max_angle, bins + 1)
        
        #already do rr  and the bin edges
        self.rr=self.rr()
        self.rr_counts,self.bin_edges = np.histogram(self.rr, bins=self.bin_array)
        self.rr_normalized = self.rr_counts / (self.n_random * (self.n_random - 1) / 2)

        ##Bin centers for plotting
        #This is actually not accurate for non linear bins!!!!!!!!
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_width = np.diff(self.bin_edges)[0]  # width of each bin


        

    def galaxy_density(self, n_galaxies):
        """
        Calculate the density of galaxies per square degree.

        Parameters
        ----------
        n_galaxies : int
            Number of galaxies in the survey.

        Returns
        -------
        density : float
            Density of galaxies per square degree.
        """
        if self.complete_sphere:
            area_sr = 4 * np.pi  # full sphere in steradians
        else:
            angle_rad= np.deg2rad(self.max_angle_plus_dr)
            #the two pi is because we are centered around the pole
            area_sr = 2 * angle_rad * (1 - np.cos(angle_rad))  # spherical cap area in steradians

        area_sqdeg = area_sr * (180 / np.pi)**2  # convert steradians to square degrees
        density = n_galaxies / area_sqdeg
        return density    
        

    
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
        unit_random=coordinate_tools.theta_phi_to_unitvec(random)
        return unit_random
    
    
    
    def tree_creation(self,coords):
        """
        Create a cKDTree from given coordinates.
        """

        ### No more periodic boundaries on the unit sphere or imcomplete sphere
        tree = ss.cKDTree(coords, boxsize=None,leafsize=self.leafsize) #optimizable?
        return tree

    
    def chord_distances_kdtree(self,tree1,tree2=None):
                    
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
        rr_chord=self.chord_distances_kdtree(tree1=self.tree_random,tree2=None)
        if self.distance_type=='angular':
            return coordinate_tools.chord_to_angular_separation(rr_chord)
        if self.distance_type=='euclidean':
            return self.radius*rr_chord
    #Call this every time we want dd or dr distances
    def dd(self,coordinates):
        d_tree=self.tree_creation(coords=coordinates)
        dd_chord=self.chord_distances_kdtree(tree1=d_tree,tree2=None)
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
        unit_coordinates=coordinate_tools.theta_phi_to_unitvec(coordinates)

        # Histogram the distances
        dd_counts, _ = np.histogram(self.dd(unit_coordinates), bins=self.bin_array)
        dr_counts, _ = np.histogram(self.dr(unit_coordinates), bins=self.bin_array)
        

        
        
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


class correlation_tools_treecorr:
    def __init__(self, cosmology,
                 min_distance=0, max_distance=250, n_random=50000,
                 max_angle=0.1, complete_sphere=True,
                 bins=100, distance_type='euclidean', seed=12345):
        
        self.n_random = n_random
        self.bins = bins
        self.distance_type = distance_type
        self.seed = seed
        self.complete_sphere=complete_sphere
        #all the relevant geometric information
        self.radius=cosmology.comoving_distance
        self.max_angle_incomplete=max_angle
        self.min_chord=min_distance/self.radius
        self.max_chord=max_distance/self.radius
        self.bao_chord=cosmology.bao_distance/self.radius
        self.bao_angle=coordinate_tools.chord_to_angular_separation(self.bao_chord)
        self.bins=bins
        
        
        
        # define bin edges and nn objects for the treecorrelation
        self.bin_edges = np.linspace(min_distance,max_distance,self.bins+1)
        self.min_sep=self.min_chord
        self.max_sep=self.max_chord
        self.bin_type='Linear'
    
            
        if distance_type == 'angular':
            #change to angular bin edges, overwriting

            #might want to get meanlogr or meanr from the treecorr module!
            self.bin_edges=coordinate_tools.chord_to_angular_separation(self.bin_edges)
            

        # bin centers for plotting
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_size = self.bin_edges[1]-self.bin_edges[0]

        # generate random catalog
        self.rng = np.random.default_rng(seed)
        self.randoms=self._generate_random()
        self.cat_random = self._catalog(self.randoms)
        
        # Precompute RR once
        self.rr = self._rr()
        

    
    
    def galaxy_density(self, n_galaxies):
        """
        Calculate the density of galaxies per square degree.

        Parameters
        ----------
        n_galaxies : int
            Number of galaxies in the survey.

        Returns
        -------
        density : float
            Density of galaxies per square degree.
        """
        if self.complete_sphere:
            area_sr = 4 * np.pi  # full sphere in steradians
        else:
            
            #the two pi is because we are centered around the pole
            # spherical cap area in steradians
            area_sr = 2 * self.max_angle * (1 - np.cos(self.max_angle))  

        area_sqdeg = area_sr * (180 / np.pi)**2  # convert steradians to square degrees
        density = n_galaxies / area_sqdeg
        return density     

    #-----------------------
    # Random catalog
    #-----------------------
    def _generate_random(self):
        """Generate random points in spherical coordinates."""
        if self.complete_sphere:
            max_cos_theta = 1.0
            max_phi = np.pi
        else:
            max_cos_theta = np.cos(self.max_angle_incomplete)
            max_phi = self.max_angle_incomplete
        
        random_theta = np.arccos(self.rng.uniform(low=-max_cos_theta, 
                                                  high=max_cos_theta,
                                                  size=self.n_random))
        random_phi = self.rng.uniform(low=-max_phi, high=max_phi, size=self.n_random)
        random = np.column_stack((random_theta, random_phi))
        return random

    #-----------------------
    # Build TreeCorr Catalogs
    #-----------------------

    
    def _catalog(self, coords_sph):
        #Secure type as float
        coords_sph = np.asarray(coords_sph, dtype=float)
    
        coords=coordinate_tools.theta_phi_to_unitvec(coords_sph)
        #Secure type as float
        coords = np.asarray(coords, dtype=float)
        return treecorr.Catalog(x=coords[:,0],
                                y=coords[:,1],
                                z=coords[:,2])
    

    #-----------------------
    # Compute RR
    #-----------------------
    def _dd(self, cat):
        nn = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.bins,
            bin_type=self.bin_type
        )
        nn.process(cat)
        return nn

    def _dr(self, cat):
        nn = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.bins,
            bin_type=self.bin_type
        )
        nn.process(cat, self.cat_random)
        return nn

    def _rr(self):
        nn = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.bins,
            bin_type=self.bin_type
        )
        nn.process(self.cat_random)
        return nn
        #-----------------------
        # Landy-Szalay estimator
        #-----------------------
    def landy_szalay(self, coords):

        #coordinates are spherical, are transformed in self._catalog
        n_data = coords.shape[0]
        cat_data = self._catalog(coords) #catalog of the coordinates, randoms is already done
        dd = self._dd(cat_data)
        dr = self._dr(cat_data)
        w_ls,var_ls=dd.calculateXi(rr=self.rr,dr=dr)
        
        return w_ls







