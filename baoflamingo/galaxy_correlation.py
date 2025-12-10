import numpy as np
import unyt as u

# pycorr imports - depending on your pycorr installation these paths may need adjusting
# The snippet you provided defines these classes inside the pycorr package; typical import paths:
from pycorr.correlation_function import TwoPointCorrelationFunction
from pycorr.twopoint_jackknife import KMeansSubsampler
# own modules
from baoflamingo.coordinates import coordinate_tools


class correlation_tools:
    """
    Reimplementation of your TreeCorr class using pycorr jackknife counters.

    Inputs:
      - coordinates: array (N, >=2). Expected format: first column redshift (z),
                     columns 1: are spherical coords (theta, phi) in radians as before.
                     In _landy_szalay we use coords[:,1:] to convert to RA/Dec same as before.
      - cosmology: your cosmo object (must provide comoving_distance_to_redshift, bao_distance, etc.)
    Outputs attached to instance after run:
      - xi_smu : 2D array (ns, nmu) measured xi(s,mu)
      - s_mid : 1D array length ns of s bin centers
      - mu_mid: 1D array length nmu of mu bin centers
      - cov   : 2D covariance matrix of flattened xi (shape (ns*nmu, ns*nmu))
      - nsamp, nmu etc available for diagnostics
    """

    def __init__(self, coordinates, cosmology, cfg, rank_id=0):
        # basic bookkeeping
        self.coordinates = coordinates
        self.n_galaxies = np.shape(coordinates)[0]
        self.n_random = int(cfg.random_catalog.oversampling * self.n_galaxies)
        self.cosmo = cosmology

        #MPI rank
        self.rank_id=rank_id
       

        
        # default s edges from min to max with `bins` linear bins
        self.s_edges = np.linspace(cfg.plotting.s_min, cfg.plotting.s_max, cfg.plotting.s_bins + 1)
        

        
        # mu in [0,1], use 10 mu bins by default (you can override)
        self.mu_edges = np.linspace(cfg.plotting.mu_min, cfg.plotting.mu_max, cfg.plotting.mu_bins+1)
        

        # derived bin mids
        self.s_bin_centers = 0.5 * (self.s_edges[:-1] + self.s_edges[1:])
        # choose mu midpoints as centers of mu bins
        self.mu_bin_centers = 0.5 * (self.mu_edges[:-1] + self.mu_edges[1:])

        # save other settings
        self.npatches = cfg.statistics.n_patches

        # RNG
        self.seed=cfg.random_catalog.seed
        self.rng = np.random.default_rng(self.seed)

        

        # Build pycorr randoms and run jackknife two-point count + estimator
        # Run the Landy-Szalay estimator using pycorr jackknife machinery
        
        self._run_pycorr(sph_coords=coordinates,cosmo=self.cosmo)
        self._survey_density()

    
    def _survey_density(self):
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
        self.survey_area = area_sr.to(u.deg**2)
        self.survey_density = self.n_galaxies /self.survey_area
        
        # calculate survey volume between inner and outer edges of the redshift bin and store it
        volume =area_sr.value * (self.cosmo.outer_edge_bin**3 - self.cosmo.inner_edge_bin**3) / 3
        self.survey_volume = volume.to(u.Gpc**3) #easier to see
        print(f"[RANK {self.rank_id}]: Percentage of total box surveyed: {(self.survey_volume/(self.cosmo.box_size**3))*100} %")
          


    def _generate_random(self,data_z):
        """Generate random points in spherical coordinates like before."""
        # use same logic as your treecorr version for incomplete sphere
        if self.cosmo.complete_sphere:
            max_phi = np.pi
            max_cos_theta = 1.0
        else:
            max_phi = self.cosmo.max_angle
            max_cos_theta = +np.sin(max_phi)

        random_theta = np.arccos(self.rng.uniform(low=-max_cos_theta,
                                                  high=max_cos_theta,
                                                  size=self.n_random))

        random_phi = self.rng.uniform(low=-max_phi,
                                      high=max_phi,
                                      size=self.n_random)


        #we have to sample using n(z) fitted by the data!!!
        # sample D_c using inverse CDF for n(z) between inner/outer edges of the data
        random_z,grid_z, n_z_norm=coordinate_tools.random_redshifts_from_data_cdf(
                                data_z=data_z,
                                n_random=self.n_random, 
                                smoothing=0.01,#Gaussian smoothing, needs tweaking!
                                rng=self.rng
                                )

        # keep same spherical coords format as your original class
        self.random = np.column_stack((random_theta,random_phi,random_z)) #(phi,theta,z) shape (N,3)
        

        # compute effective redshift/radius for geometric conversions (optional)
        #this ultimately decides the tension at that redshift
        self.effective_redshift = self.cosmo.effective_redshift(random_z)
        self.effective_radius = self.cosmo.effective_comoving_distance(random_z)
        self.effective_H_z=self.cosmo.effective_hubble_constant(random_z)
        self.effective_D_a=self.cosmo.effective_angular_diameter_distance(random_z)



    def _run_pycorr(self, sph_coords,cosmo):
        """
        Build pycorr two-point counts with jackknife and produce xi(s,mu) and covariance.
        `catalog_coords` is expected as in your original code: first col redshift, then (theta,phi)
        
        Always input the data an randoms in the form (2,N_points) for (ra,dec) 
        or (3,N_points) for cartesian coordinates.

        """

        # --- Generate randoms (in spherical coords as before)
        self._generate_random(sph_coords[:,2])#input z coordinates of data to get cdf distribution

        # ---- Prepare data positions (cartesian) using a  cosmology ----
        
        # data and randoms are in the form (N,(theta,phi,z)) and going to ((ra,dec,r),N)

        """
        print(f"maximum range of theta in data: {np.max(sph_coords[:,0])}, minimum: {np.min(sph_coords[:,0])} ")
        print(f"maximum range of theta in randoms: {np.max(self.random[:,0])}, minimum: {np.min(self.random[:,0])}")
        print(f"maximum range of phi in data: {np.max(sph_coords[:,1])}, minimum: {np.min(sph_coords[:,1])}")
        print(f"maximum range of phi in randoms: {np.max(self.random[:,1])}, minimum: {np.min(self.random[:,1])}")
        """
        pos_data,pos_units = coordinate_tools.theta_phi_z_to_ra_dec_r(sph_coords, cosmo)# shape (3,N)
        pos_random,__ =coordinate_tools.theta_phi_z_to_ra_dec_r(self.random, cosmo)# shape (3,N)
        
        # ---- Build jackknife subsampler on the randoms (ra,dec) coordinates;
        # using kmeans for roughly equal-area patches ---- 
     
       
        subsampler = KMeansSubsampler(mode='angular', positions=pos_random[:2,:], 
                nsamples=self.npatches,position_type='rd', random_state=self.seed)
        
        # labels for each random (and data) point:
        patch_labels_random = subsampler.label(pos_random[:2,:], position_type='rd')
        patch_labels_data = subsampler.label(pos_data[:2,:], position_type='rd')
        
       

        # ---- Setup edges for pycorr JackknifeTwoPointCounter ----
        # pycorr expects tuple of arrays; for mode 'smu' that's (s_edges, mu_edges)

        self.edges = (self.s_edges, self.mu_edges)  
        # For weight handling, we do not provide weights (None)
        weights_data = None
        weights_random = None

        # Now instantiate the TwoPointCorrelationFunction
  
        ls = TwoPointCorrelationFunction(
            mode='smu',
            #data
            data_positions1=pos_data,
            data_samples1=patch_labels_data,
            data_weights1=weights_data,
            #randoms
            randoms_positions1=pos_random,
            randoms_samples1=patch_labels_random,
            randoms_weights1=weights_random,
            #settings
            bin_type='lin',          # linear bins -> faster
            position_type='rdd',     # positions are provided as (3,N)
            edges=self.edges,
            #weight_type=None, #this is acting weird
            los='midpoint',
            estimator='landyszalay',
            nthreads=1,
            gpu=False
            )
        #actually running the estimator (including the jackknife method)
        xi,cov=ls.get_corr(return_cov=True)
       # Boolean array where True indicates NaN
        nan_mask = np.isnan(xi)

        # Count total number of NaNs
        num_nans = np.sum(nan_mask)
        if num_nans>0:
            print(f"[RANK {self.rank_id}] Number of bins that are empty:", num_nans)
        
        self.xi= xi #(ns, nmu) array
        self.cov= cov #(ns*nmu, ns*nmu) array
        # keep metadata
        self.patch_labels_data = patch_labels_data
        self.patch_labels_randoms = patch_labels_random
