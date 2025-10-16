import numpy as np
import treecorr
import unyt as u
#own modules
from baoflamingo.coordinates import coordinate_tools




class correlation_tools_treecorr:
    def __init__(self, cosmology,
                 min_distance=0, max_distance=250, n_random=50000,
                 max_angle=0.1, complete_sphere=True,
                 bins=100, distance_type='euclidean', seed=12345,
                 variance_method='jackknife',n_patches=100):
        
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
        self.bao_angle=coordinate_tools.chord_to_angular_degrees(self.bao_chord)*u.deg
        self.bins=bins
        
        
        
        # define bin edges and nn objects for the treecorrelation
        self.min_sep=self.min_chord
        self.max_sep=self.max_chord
        self.bin_type='Linear' 
            
        
        
        

        # generate random catalog
        self.rng = np.random.default_rng(seed)
        self.randoms=self._generate_random()

        #first set patches at a number previously set up
        self.npatches=n_patches
        self.patch_centers=None
        self.cat_random = self._catalog(self.randoms)
        # Precompute RR once
        self.rr = self._rr()
        
        #now set up patch centers as mentioned before and get npatches=None
        #only done when n_patches is defined!!!
        #also define the method used
        #cpw=Cross-patch Weights, see TreeCorr documentation
        self.variance_method=variance_method
        if self.variance_method=='jackknife':
            self.cpw='match'
        elif self.variance_method=='bootstrap':
            self.cpw='geom'
        else:
            self.cpw='simple'

        #this if for the data catalogue, resetting 
        self.patch_centers=self.cat_random.patch_centers
        
        
        
        

    
    
    def galaxy_density(self, n_galaxies):
        """
        Calculate the density of galaxies per square degree, with units.

        Parameters
        ----------
        n_galaxies : int
            Number of galaxies in the survey.

        Returns
        -------
        density : astropy.units.Quantity
            Density of galaxies per square degree.
        """
        if self.complete_sphere:
            area_sr = 4 * np.pi * u.sr  # full sphere in steradians
        else:
            # Spherical cap area in steradians
            area_sr = 2 * self.max_angle_incomplete * (1 - np.cos(self.max_angle_incomplete)) * u.sr
        
        # Convert steradians to square degrees
        area_sqdeg = area_sr.to(u.deg**2)
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

        """
        #Secure type as float
        coords_sph = np.asarray(coords_sph, dtype=float)
        #overwriting=memory efficiency 
        coords_sph=coordinate_tools.theta_phi_to_ra_dec(coords_sph) #ra,dec in degrees
        #Secure type as float, double check
        coords_sph = np.asarray(coords_sph, dtype=float)


        if self.patch_centers is None: #for the random catalogue patch_centers is empty
            return treecorr.Catalog(ra=coords_sph[:,0],
                                    dec=coords_sph[:,1],
                                    npatch=self.npatches,
                                    
                                    ra_units='degrees',
                                    dec_units='degrees'
                                    )
        else: #for the data catalogue
            return treecorr.Catalog(ra=coords_sph[:,0],
                                    dec=coords_sph[:,1],
                                    patch_centers=self.patch_centers,
                                    
                                    ra_units='degrees',
                                    dec_units='degrees'
                                    )
        """
        coords_sph = np.asarray(coords_sph, dtype=float)
        #overwriting=memory efficiency 
        coords=coordinate_tools.theta_phi_to_unitvec(coords_sph) #x,y,z
        #Secure type as float, double check
        coords = np.asarray(coords, dtype=float)

        """
        print(np.max(coords_sph[:,0]),np.max(coords_sph[:,1]))
        print(np.min(coords_sph[:,0]),np.min(coords_sph[:,1]))
        print(np.average(coords_sph[:,0]),np.average(coords_sph[:,1]))
        """
        if self.patch_centers is None: #for the random catalogue patch_centers is empty
            return treecorr.Catalog(x=coords[:,0],
                                    y=coords[:,1],
                                    z=coords[:,2],
                                    npatch=self.npatches,
                                    
                                    )
        else: #for the data catalogue
            return treecorr.Catalog(x=coords[:,0],
                                    y=coords[:,1],
                                    z=coords[:,2],
                                    patch_centers=self.patch_centers,

                                    )

                                
    

    #-----------------------
    # Compute RR
    #-----------------------
    def _dd(self, cat):
        nn = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.bins,
            bin_type=self.bin_type,
            var_method=self.variance_method,
            cross_patch_weight=self.cpw,
            metric='Euclidean'

        )
        nn.process(cat)
        return nn

    def _dr(self, cat):
        nn = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.bins,
            bin_type=self.bin_type,
            metric='Euclidean'
        )
        nn.process(cat, self.cat_random)
        return nn

    def _rr(self):
        nn = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.bins,
            bin_type=self.bin_type,
            metric='Euclidean'
        )
        nn.process(self.cat_random)
        return nn
        #-----------------------
        # Landy-Szalay estimator
        #-----------------------
    def landy_szalay(self, coords):

        #coordinates are spherical, are transformed in self._catalog to ra,dec
        #catalog of the coordinates, randoms is already done
        cat_data = self._catalog(coords) 
        dd = self._dd(cat_data)
        dr = self._dr(cat_data)
        dd.calculateXi(rr=self.rr,dr=dr)
        mean=dd.xi
        std=dd.varxi
        

        #for now only using the diagonal elements, later on maybe using the covariances!!!
        
        
        chord_centers=dd.rnom #in chord distance
        if self.distance_type == 'angular':
            #might want to get meanlogr or meanr from the treecorr module!
            self.bin_centers=(coordinate_tools.chord_to_angular_degrees(chord_centers))*u.deg

        if self.distance_type == 'euclidean':
            self.bin_centers=(chord_centers*self.radius)*u.Mpc #actual euclidean distances
        #not exactly true for the angular case, only relatively true for small angles
        self.bin_width = self.bin_centers[1]-self.bin_centers[0]
        
        
        return mean,std

    








