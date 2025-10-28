import numpy as np
import treecorr
import unyt as u
#own modules
from baoflamingo.coordinates import coordinate_tools






class correlation_tools_treecorr:
    def __init__(self, 
                 coordinates,
                 cosmology,
                 min_distance=0, max_distance=250, n_random=50000,
                 bins=10, distance_type='euclidean', seed=12345,
                 variance_method='jackknife',n_patches=100):
        
        self.n_galaxies=np.shape(coordinates)[0]
        self.n_random = n_random
        self.bins = bins
        self.distance_type = distance_type
        self.seed = seed
        
        
        #all the relevant geometric information inside cosmology instance
        self.cosmo=cosmology
        self.min=min_distance
        self.max=max_distance
        self.bao=self.cosmo.bao_distance
        
        # define bin edges and nn objects for the treecorrelation
        self.bin_type='Linear' 
       

        #This is a question on how we define the effective redshift and comoving distance
        #either base this on the datapoints or randoms!!!
        self.min_distance=min_distance
        self.max_distance=max_distance
        """ 
        #effective redshift and comoving distance inside the redshift bin
        self.effective_redshift=self.cosmo.effective_redshift(coordinates[:,0])
        self.effective_radius=self.cosmo.effective_comoving_distance(coordinates[:,0])
        
        #Limits for the w(theta) calculation
        min_chord=min_distance/self.effective_radius
        max_chord=max_distance/self.effective_radius
        bao_chord=self.cosmo.bao_distance/self.effective_radius
        self.min_rad=coordinate_tools.chord_to_angular_radians(min_chord)
        self.max_rad=coordinate_tools.chord_to_angular_radians(max_chord)
        self.bao_rad=coordinate_tools.chord_to_angular_radians(bao_chord)
        """ 

        # generate random catalog
        self.rng = np.random.default_rng(seed)
        #generate random redshifts and theta_phi
        self._generate_random()


        #--- Angular correlation here ---
        
        #first set patches at a number previously set up
        self.npatches=n_patches
        self.patch_centers=None
        self.cat_random = self._catalog(self.randoms_theta_phi)
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

        self.ls_avg,self.ls_std=self._landy_szalay(coordinates)
        self._galaxy_density() #density and area are added
        
        
        
        

    
    
    def _galaxy_density(self):
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
        if self.cosmo.complete_sphere:
            area_sr = 4 * np.pi * u.sr  # full sphere in steradians
        else:
            # Spherical cap area in steradians
            area_sr = 2 * self.cosmo.max_angle.value \
            * (1 - np.cos(self.cosmo.max_angle.value)) * u.sr
        
        # Convert steradians to square degrees
        self.observed_area = area_sr.to(u.deg**2)
        self.galaxy_density = self.n_galaxies /self.observed_area
          

    #-----------------------
    # Random catalog
    #-----------------------
    def _generate_random(self):
        """Generate random points in spherical coordinates and set up geometric properties"""
        if self.cosmo.complete_sphere:
            max_phi = np.pi
            max_cos_theta= 1.0
            
        else:
            max_phi = self.cosmo.max_angle
            max_cos_theta = +np.sin(max_phi)
            
        ##we have to look at this!
        random_theta =np.arccos(self.rng.uniform(low=-max_cos_theta, 
                                                  high=max_cos_theta,
                                                  size=self.n_random))
        
        random_phi = self.rng.uniform(low=-max_phi,
                                      high=max_phi,
                                      size=self.n_random)
        #sample D_c using inverse CDF for p(D_c) ∝ D_c^2
        u = self.rng.random(self.n_random)
        D_c = ((u * (self.cosmo.outer_edge_bin**3 - self.cosmo.inner_edge_bin**3)) \
                    + self.cosmo.inner_edge_bin**3)**(1/3)
        
        #linking redshift to D_c
        self.randoms_redshift=self.cosmo.comoving_distance_to_redshift(D_c.value)
        self.randoms_theta_phi = np.column_stack((random_theta, random_phi))


        # ---- GEOMETRIC EFFECTIVE REDSHIFT AND COMOVING DISTANCE ---
        #effective redshift and comoving distance inside the redshift bin
        self.effective_redshift=self.cosmo.effective_redshift(self.randoms_redshift)
        self.effective_radius=self.cosmo.effective_comoving_distance(self.randoms_redshift)
        print(f"z_eff={self.effective_redshift}")
        print(f"d_c_eff={self.effective_radius}")
        
        #Limits for the w(theta) calculation
        min_chord=self.min_distance/self.effective_radius
        max_chord=self.max_distance/self.effective_radius
        bao_chord=self.cosmo.bao_distance/self.effective_radius
        self.min_rad=coordinate_tools.chord_to_angular_radians(min_chord)
        self.max_rad=coordinate_tools.chord_to_angular_radians(max_chord)
        self.bao_rad=coordinate_tools.chord_to_angular_radians(bao_chord)
        
        

    #-----------------------
    # Build TreeCorr Catalogs for the ra,dec angular correlation part
    #-----------------------

    
    def _catalog(self, coords):

        
        #overwriting=memory efficiency, ra,dec in radians 
        coords=coordinate_tools.theta_phi_to_ra_dec(coords) 
        

        #This is a check for the incomplete sphere if the boundaries (theta,phi)
        #for the data and randoms is the same!!!!
        """
        print("At _catalog() entry, shape:", coords_sph.shape)
        
        print("Theta col min/max:", coords_sph[:,0].min(), coords_sph[:,0].max())
        print("Phi col min/max:", coords_sph[:,1].min(), coords_sph[:,1].max())
        """
        
        
        #for the random catalogue patch_centers is empty, initialize with the npatches
        if self.patch_centers is None: 
            return treecorr.Catalog(ra=coords[:,0],
                                    dec=coords[:,1],
                                    ra_units='radians',
                                    dec_units='radians',
                                    npatch=self.npatches,
                                    
                                    )
        else: #for the data catalogue, use patch centers
            return treecorr.Catalog(ra=coords[:,0],
                                    dec=coords[:,1],
                                    ra_units='radians',
                                    dec_units='radians',
                                    patch_centers=self.patch_centers,

                                    )

                                
    

    #-----------------------
    # Compute RR
    #-----------------------
    def _dd(self, cat):
        nn = treecorr.NNCorrelation(
            min_sep=self.min_rad.value,
            max_sep=self.max_rad.value,
            sep_units='radians',
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
            min_sep=self.min_rad.value,
            max_sep=self.max_rad.value,
            sep_units='radians',
            nbins=self.bins,
            bin_type=self.bin_type,
            metric='Euclidean'
        )
        nn.process(cat, self.cat_random)
        return nn

    def _rr(self):
        nn = treecorr.NNCorrelation(
            min_sep=self.min_rad.value,
            max_sep=self.max_rad.value,
            sep_units='radians',
            nbins=self.bins,
            bin_type=self.bin_type,
            metric='Euclidean'
        )
        nn.process(self.cat_random)
        return nn
        #-----------------------
        # Landy-Szalay estimator
        #-----------------------
    def _landy_szalay(self, coords):

        #coordinates are spherical, are transformed in self._catalog to ra,dec
        #catalog of the coordinates, randoms is already done

        #IMPLEMENT LOS CORRELATION
        

        cat_data = self._catalog(coords[:,1:]) #only (theta,phi) needed
        dd = self._dd(cat_data)
        dr = self._dr(cat_data)
        dd.calculateXi(rr=self.rr,dr=dr)
        mean=dd.xi
        std=dd.varxi
        

        #for now only using the diagonal elements, later on maybe using the covariances!!!
        
        
        radians_centers=dd.rnom #in radians
        if self.distance_type == 'angular':
            #might want to get meanlogr or meanr from the treecorr module!
            #overwriting the euclidean distances with angular distances in rad
            self.bin_centers=radians_centers*u.rad
            self.bao=self.bao_rad
            self.min=self.min_rad
            self.max=self.max_rad
        
        if self.distance_type == 'euclidean':
            chord_centers=coordinate_tools.angular_radians_to_chord(radians_centers)
            self.bin_centers=(chord_centers*self.effective_radius)
        #not exactly true for the angular case, only relatively true for small angles
        self.bin_width = self.bin_centers[1]-self.bin_centers[0]
        
        
        return mean,std

    