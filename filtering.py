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


    def _luminosity_filter(self,file):

        return mask
