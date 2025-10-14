import numpy as np
import unyt as u
from galaxy_correlation import cosmo_tools
from scipy.interpolate import interp1d


class filtering_tools:
    def __init__(self,soap_file,cosmology,
    central_filter=False,
    stellar_mass_filter=False,stellar_mass_cutoff=0,
    luminosity_filter=False, filter_band='r',m_cutoff=22.0):
        
        
        #saving cosmology instance
        self.cosmo=cosmology
        
        #stellar mass filter parameters
        self.stellar_mass_cutoff=stellar_mass_cutoff
        #luminosity filter parameters
        self.filter_band=filter_band
        self.m_cutoff=m_cutoff
        

        #empty mask if filters are not applied
        self.mask_empty=np.ones(
            len(soap_file.bound_subhalo.stellar_mass),dtype=bool)

        #Here we decide which filters to apply
        self.central_mask=self._central_filter(soap_file) if central_filter else self.mask_empty
        self.stellar_mass_mask=self._stellar_mass_filter(soap_file) if stellar_mass_filter  else self.mask_empty
        self.luminosity_mask=self._luminosity_filter(soap_file) if luminosity_filter else self.mask_empty
        #combine all masks but leave out the radial mask, which has to be applied later
        #as it also depends on the coordinates in the main file
        self.total_mask=self.central_mask & self.stellar_mass_mask & self.luminosity_mask



    
    def radial_filter(self,coordinates):
       
        r = coordinates[:, 0]
        mask = (r >= self.cosmo.minus_dr) & (r <= self.cosmo.plus_dr)
        #only send back theta and phi
        return coordinates[mask][:, 1:]

    def _central_filter(self,file):
        
        mask=file.input_halos.is_central
        return (mask)

    
    def _stellar_mass_filter(self,file):
        stellar_mass=file.bound_subhalo.stellar_mass
        mask= (stellar_mass>=stellar_mass_cutoff)
        return (mask)


    def _luminosity_filter(self,file):
        #Effective wavelengths of bands in nm
       
        bands=['u','g','r','i','z','Y','J','H','K']
        lambda_eff=[354,475,622,763,905,1031,1248,1631,2201] #in nm
        log_lambda_eff=np.log10(lambda_eff)

        lum = file.bound_subhalo.stellar_luminosity.value  # shape (N_gal, N_bands)

        # Filter out galaxies with any zero luminosity (or just in the u-band)
        nonzero_mask = lum[:, bands.index('u')] != 0
        lum = lum[nonzero_mask, :]

        #loading in luminosities in all bands
        #we have to convert to absolute magnitudes in each band
        M_ab_bands=-2.5*np.log10(lum) #in AB mag
        
        #calculate rest frame band from input band and redshift
        log_rest_band=np.log10(lambda_eff[bands.index(self.filter_band)]/(1+self.cosmo.redshift))

        

        #Now interpolate to find the rest frame band absolute magnitude
         # Interpolate in log-log space
         #use scipy for interpolation, much faster than numpy

        
        interp_func = interp1d(
                    log_lambda_eff,
                    M_ab_bands.T,  # shape (N_bands, N_galaxies), transposed
                    kind='linear',
                    axis=0,
                    bounds_error=False,
                    fill_value='extrapolate'
                )

        M_ab_rest = interp_func(log_rest_band)  # returns (N_galaxies,)


        #calculate apparent magnitude in the selected band
        D_L = self.cosmo.luminosity_distance *1e6 # convert Mpc → pc
        m = M_ab_rest + 5 * np.log10(D_L)  -5

        mask_m=(m<=self.m_cutoff)


        # Map back to full catalog length
        #combine with zero luminosity mask to send back total mask
        full_mask = np.zeros(len(nonzero_mask), dtype=bool)
        full_mask[nonzero_mask] = mask_m
        
        pass_fraction = np.count_nonzero(full_mask) / full_mask.size * 100
        print(f"Pass percentage after luminosity filter: {pass_fraction:.2f}%")



        #memory clean up of large arrays
        del lum,M_ab_bands, m, M_ab_rest
        
        return full_mask 