import numpy as np
import unyt as u


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

        

        abs_mag_bands=-2.5*np.log10(file.bound_subhalo.steller_luminosity.value)

        #we need to apply k-correction here

        #we have to decide which band to use for the apparent magnitude cutoff
        band_index=bands.index(band)

        #we need to decide the rest frame band of each galaxy with interpolation
        #we will use interpolation to find the luminosity in the rest frame band
        #we will use the two closest bands to the redshifted band
        #we will use linear interpolation in magnitude space
        
        #then we apply the distance modulus to get the apparent magnitude
        #m = M + 5 * log10(D_L / 10 pc)
        #where D_L is the luminosity distance in parsecs
        #we can use astropy to get the luminosity distance
        return mask
