import numpy as np
import unyt as u

# pycorr imports - depending on your pycorr installation these paths may need adjusting
# The snippet you provided defines these classes inside the pycorr package; typical import paths:
from pycorr.correlation_function import TwoPointCorrelationFunction
from pycorr.twopoint_jackknife import KMeansSubsampler
# own modules
from baoflamingo.coordinates import coordinate_tools


class correlation_tools_pycorr:
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

    def __init__(self,
                 coordinates,
                 cosmology,
                 cfg):
        # basic bookkeeping
        self.coordinates = coordinates
        self.n_galaxies = np.shape(coordinates)[0]
        self.n_random = int(cfg.random_catalog.oversampling * self.n_galaxies)
        self.seed = cfg.random_catalog.seed
        self.cosmo = cosmology
       

        
        # default s edges from min to max with `bins` linear bins
        self.s_edges = np.linspace(cfg.plotting.s_min, cfg.plotting.s_max, cfg.plotting.s_bins + 1)
        

        
        # mu in [0,1], use 10 mu bins by default (you can override)
        self.mu_edges = np.linspace(cfg.plotting.mu_min, cfg.plotting.mu_max, cfg.plotting.mu_bins+1)
        

        # derived bin mids
        self.s_bins_center = 0.5 * (self.s_edges[:-1] + self.s_edges[1:])
        # choose mu midpoints as centers of mu bins
        self.mu_bins_center = 0.5 * (self.mu_edges[:-1] + self.mu_edges[1:])

        # save other settings
        self.npatches = cfg.statistics.n_patches

        # RNG
        self.rng = np.random.default_rng(self.seed)

        # generate randoms (in spherical coords as before)
        self._generate_random()

        # Build pycorr randoms and run jackknife two-point count + estimator
        # Run the Landy-Szalay estimator using pycorr jackknife machinery
        self._run_pycorr(catalog_coords=coordinates)

    def _generate_random(self):
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

        # sample D_c using inverse CDF for p(D_c) ∝ D_c^2 between inner/outer edges
        u_rand = self.rng.random(self.n_random) #0 to 1
        D_c = ((u_rand * (self.cosmo.outer_edge_bin**3 - self.cosmo.inner_edge_bin**3)) \
               + self.cosmo.inner_edge_bin**3)**(1/3)

        # map to redshift using your existing cosmology interface
        random_redshift = self.cosmo.comoving_distance_to_redshift(D_c.value)

        # keep same spherical coords format as your original class
        self.random = np.column_stack((random_redshift, random_theta, random_phi)) #z,theta,phi shape (N,3)
        

        # compute effective redshift/radius for geometric conversions (optional)
        self.effective_redshift = self.cosmo.effective_redshift(self.randoms_redshift)
        self.effective_radius = self.cosmo.effective_comoving_distance(self.randoms_redshift)



    def _run_pycorr(self, sph_coords,cosmo):
        """
        Build pycorr two-point counts with jackknife and produce xi(s,mu) and covariance.
        `catalog_coords` is expected as in your original code: first col redshift, then (theta,phi)
        
        Always input the data an randoms in the form (2,N_points) for (ra,dec) 
        or (3,N_points) for cartesian coordinates.
        """

        # ---- Prepare data positions (cartesian) using a  cosmology ----
        
        # data and randoms are in the form (N,(z,theta,phi))
        pos_data = (coordinate_tools.spherical_to_cartesian_positions(sph_coords, cosmo)).T  # shape (3,N)
        # --- introduce ra_dec for data to introduce patches for jackknifing, shape (2,N)
        ra_dec_data= (coordinate_tools.theta_phi_to_ra_dec(sph_coords[:,:1]).to('deg')).T#convert to degrees
        
        # ---- Prepare random positions converted using the same cosmology (for RR) ----
    
        pos_random = (coordinate_tools.spherical_to_cartesian_positions(self.random,cosmo)).T # shape (3,N)
        # --- introduce ra_dec for randoms to introduce patches for jackknifing, shape (2,N)
        ra_dec_random= coordinate_tools.theta_phi_to_ra_dec(self.random[:,:1]).to('deg')#convert to degrees
        
        
        # ---- Build jackknife subsampler on the randoms (ra,dec) coordinates;
        # using kmeans for roughly equal-area patches ---- 
     
        
        subsampler = KMeansSubsampler(mode='angular', positions=ra_dec_random, 
                nsamples=self.npatches,position_type='rd', random_state=self.seed)
        # labels for each random (and data) point:
        patch_labels_random = subsampler.label(ra_dec_random, position_type='rd')
        patch_labels_data = subsampler.label(pos_data, position_type='rd')
       

        # ---- Prepare pycorr inputs: positions as arrays of shape (3, N) ----
        # pycorr expects a list/array of 3 arrays (x,y,z) for positions depending on position_type; we will pass 'pos' in TwoPointCounter so use pos.T
        positions1 = pos_data.T  # shape (3, Ndata)
        positions2 = None  # maybe a shifted 
        # samples1: labels of jackknife region for each galaxy
        samples1 = patch_labels_data
        samples2 = None

        # ---- Setup edges for pycorr JackknifeTwoPointCounter ----
        # pycorr expects tuple of arrays; for mode 'smu' that's (s_edges, mu_edges)

        self.edges = (self.s_edges, self.mu_edges)  
        # For weight handling, we do not provide weights (None)
        weights_data = None
        weights_random = None

        # Now instantiate the JackknifeTwoPointCounter to compute jackknife realizations
        # NOTE: the exact constructor call can vary across pycorr versions. The snippet you provided indicates:
        # JackknifeTwoPointCounter(mode, edges, positions1, samples1, weights1=..., positions2=..., samples2=..., ...)
        
        ls = TwoPointCorrelationFunction(
            mode='smu',
            edges=self.edges,
            data_positions1=pos_data,
            data_samples1=patch_labels_data,
            data_weights1=weights_data,
            randoms_positions1=pos_random,
            randoms_samples1=patch_labels_random,
            randoms_weights1=weights_random,
            bin_type='lin',          # linear bins -> faster
            position_type='xyz',     # positions are provided as (3,N) pos
            weight_type=None,
            los='midpoint',
            estimator='landayszalay',
            nthreads=1,
            )

        dir(ls)
        # store outputs on the instance
        self.ls= ls
        # keep metadata
        self.patch_labels_data = patch_labels_data
        self.patch_labels_randoms = patch_labels_randoms
