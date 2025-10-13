import numpy as np
import unyt as u
from galaxy_correlation import cosmo_tools


class FilteringTools:
    def __init__(self,soap_file,min_radius,max_radius,redshift,
    central_filter=False, stelar_mass_filter=False,luminosity_filter=False):
        self.min_radius=min_radius
        self.max_radius=max_radius
        
        self.central_mask=_central_filter(soap_file)
        self.gama_bands=gama_bands  
        
        #we will apply them to the functions, we do not have to save them as attributes !!!!!
        self.redshift=redshift
        self.stellar_mass_cutoff=stellar_mass_cutoff
        self.mr_cutoff=mr_cutoff
        

        #empty mask if filters are not applied
        self.mask_empty=np.ones(len(soap_file.bound_halos.stellar_mass),dtype=bool)

        #Here we decide which filters to apply
        self.central_mask=self._central_filter(soap_file) if central_filter else self.mask_empty
        self.stellar_mass_mask=self._stellar_mass_filter(soap_file) if stellar_mass_filter  else self.mask_empty
        self.luminosity_mask=self._luminosity_filter(soap_file) if luminosity_filter else self.mask_empty




    
    def radial_filter(self,coordinates):
       
        r = coordinates[:, 0]
        mask = (r >= self.min_radius) & (r <= self.max_radius)
        return mask

    def _central_filter(self,file):
        
        mask=file.input_halos.is_central
        return (mask)

    
    def _stellar_mass_filter(self,file,cutoff)
        stellar_mass=self.file.bound_subhalo.stellar_mass
        mask=stellar_mass>=cutoff
        return (mask)


    def _luminosity_filter(self,file,m_cutoff,redshift,band):
        
        bands=['u','g','r','i','z','Y','J','H','K']
        lambda_eff=[354,475,622,763,905,1031,1248,1631,2201] #in nm
        log_lambda_eff=np.log10(lambda_eff)



        #loading in luminosities in all bands
        #we have to convert to absolute magnitudes in each band
        M_ab_bands=-2.5*np.log10(file.bound_subhalo.steller_luminosity.value) #in AB mag
        obs_band=band
        log_rest_band=np.log10(obs_band/(1+redshift))

        #Check if there are any zeroes in luminosity to filter out
        M_ab_u=M_ab_bands[:,bands.index('u')]
        zero_mask_u= (M_ab_u!=np.inf)
        #overwrite M_ab_bands to remove zero luminosity galaxies
        #dont forget to send the total mask including the zero_mask_u
        M_ab_bands=M_ab_bands[zero_mask_u,:]

        #Now interpolate to find the rest frame band absolute magnitude
         # Interpolate in log-log space
        M_ab_rest = np.interp(log_rest_band, log_lambda_eff, M_ab_bands)
        
        

        #calculate apparent magnitude in the selected band
        m=cosmo_tools.apparent_magnitude(
            absolute_magnitude=M_ab_rest,
            z=redshift)

        mask_m=(m<=m_cutoff)

        #combine with zero luminosity mask to send back total mask
        return (mask_m & zero_mask_u)