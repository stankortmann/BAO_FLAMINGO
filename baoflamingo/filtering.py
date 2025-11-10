import numpy as np
import unyt as u
from baoflamingo.cosmology import cosmo_tools
from scipy.interpolate import interp1d


class filtering_tools:
    def __init__(self,soap_file,cosmology,cfg):
        
        #saving the soap file inside the instance, might be good to remain outside but not quite sure
        self.file=soap_file

        #saving cosmology instance
        self.cosmo=cosmology

        #stellar mass filter parameters and if we have to use it
        self.stellar_mass_filter_switch=cfg.filters.stellar_mass_filter
        self.stellar_mass_cutoff=cfg.filters.stellar_mass_cutoff
        #luminosity filter
        self.survey=cfg.filters.survey 
        self.luminosity_filter_switch=cfg.filters.luminosity_filter
        #only in case of bgs
        if self.survey=='bgs':
            self.m_r_cutoff=cfg.filters.m_r_cutoff
        
        
        #empty mask if filters are not applied, complete pass
        self.empty_mask=np.ones(
            len(soap_file.bound_subhalo.stellar_mass),dtype=bool)
        


        #We always update these filters so you can apply them when calling a filter
        #this one will keep track of what galaxies have already been masked
        self.total_mask=self.empty_mask

        #this one will actually be applied onto the coordinates of the haloes
        self.apply_mask=self.empty_mask 

        




    
    def radial_filter(self,coordinates):
        old_mask=self.total_mask.copy() #we do not apply this to coordinates,
        #is already done in the pipeline!
        
        r = coordinates[:, 0]
        radial_mask = (r >= self.cosmo.inner_edge_bin) & (r <= self.cosmo.outer_edge_bin)
        complete_mask=radial_mask

        
        if self.cosmo.complete_sphere ==False:
            #extra boundaries have to be set!!
            #phi boundaries
            phi=coordinates[:,2]
            max_phi = self.cosmo.max_angle.value
            phi_mask = (phi >= -max_phi) & (phi <= max_phi) 
            complete_mask &=phi_mask

            #theta boundaries
            theta=coordinates[:,1]
            max_cos_theta = np.sin(max_phi)
            min_theta = np.arccos(max_cos_theta)
            max_theta = np.arccos(-max_cos_theta)
            theta_mask = (theta >= min_theta) & (theta <=max_theta)
            complete_mask &= theta_mask
        
        sph_coordinates=coordinates[complete_mask] #(r,theta,phi)
        rad_coordinates=sph_coordinates[:,0] #(r)
        # r ---> z
        redshift_coordinates=self.cosmo.comoving_distance_to_redshift(rad_coordinates)
        # now introduce an error in the z_coordinate and overwrite
        redshift_coordinates=self.cosmo.redshift_with_error(redshift_coordinates)
        #might need extra redshift filtering after this!!
        #maybe revert back to r?
        #now unpack the theta and phi within 
        theta_phi_coordinates=sph_coordinates[:,1:]

        #updating internal total_mask
        old_mask[old_mask]=complete_mask
        self.total_mask=old_mask
        
        #final stacking
        complete_coordinates=np.column_stack((theta_phi_coordinates,redshift_coordinates))

        print("Radial filter applied")
        return complete_coordinates #(theta,phi,z)

    def redshift_filter(self,coordinates): #input is (theta,phi,z)
        """
        Extra filter to get only the galaxies in the appointed redshiftbin
        """
        old_mask=self.total_mask.copy()
        z=coordinates[:,2]
        redshift_mask= (z >= self.cosmo.min_redshift) & (z <= self.cosmo.max_redshift)
        
        #updating internal total_mask
        old_mask[old_mask]=redshift_mask
        self.total_mask=old_mask
        
        coordinates=coordinates[redshift_mask]
        return coordinates


    def central_filter(self,coordinates):
        old_mask=self.total_mask.copy()

        central_mask=self.file.input_halos.is_central.value[old_mask]
        #updating internal total_mask
        old_mask[old_mask]=central_mask
        self.total_mask=old_mask
        
        coordinates=coordinates[central_mask]
        print("Central filter applied")
        return coordinates

    
    def stellar_mass_filter(self,coordinates):
        old_mask=self.total_mask.copy()

        stellar_mass=self.file.bound_subhalo.stellar_mass[old_mask]
        stellar_mass_mask= (stellar_mass>self.stellar_mass_cutoff)
        #updating internal total_mask
        old_mask[old_mask]=stellar_mass_mask
        self.total_mask=old_mask
        
        coordinates=coordinates[stellar_mass_mask]
        print("Stellar mass filter applied")
        return coordinates


    def zero_luminosity_filter(self,coordinates):
        old_mask=self.total_mask.copy()
        
        lum = self.file.bound_subhalo.stellar_luminosity.value[old_mask]
        mask_zero = np.any(lum == 0, axis=1)
        #updating internal total_mask
        old_mask[old_mask] = ~mask_zero
        self.total_mask=old_mask
        
        coordinates=coordinates[~mask_zero]
        return coordinates


    def luminosity_filter(self,coordinates):
        old_mask=self.total_mask.copy()
        
        #Effective wavelengths of bands in nm
       
        bands=['u','g','r','i','z','Y','J','H','K']
        lambda_eff=np.array([354,475,622,763,905,1031,1248,1631,2201]) #in nm
        log_lambda_eff=np.log10(lambda_eff)
        
        # shape (N_gal, N_bands)
        lum = self.file.bound_subhalo.stellar_luminosity.value[old_mask]
        
        
        
        # Filter out galaxies with any zero luminosity (or just in the u-band)
        #now already done with the stellar mass filter

        #loading in luminosities in all bands
        #we have to convert to absolute magnitudes in each band
        M_ab_bands=-2.5*np.log10(lum) #in AB mag
        

        

        #Now interpolate to find the rest frame band absolute magnitude
         # Interpolate in log-log space
         #use scipy for interpolation, much faster than numpy

        """
        interp_func = interp1d(
                    log_lambda_eff,
                    M_ab_bands.T,  # shape (N_bands, N_galaxies), transposed
                    kind='linear',
                    axis=0,
                    bounds_error=False,
                    fill_value='extrapolate'
                )
        """

        


        #calculate apparent magnitude in the selected band
        #use the z component of the coordinates to calculate the luminosity distance
        D_L = self.cosmo.luminosity_distance(coordinates[:,2]).to('pc')# convert Mpc → pc)
        
        """
        print('statistics of apparent magnitude:')
        print('maximum:',np.max(m))
        print('minimum:',np.min(m))
        print('average:',np.average(m))
        print('median:',np.median(m))
        """

        def band_apparent_magnitude(z,band):
            #calculate rest frame band from input band and redshift

            if band=='w1':
                #3368nm is the lambda eff for the w1 filter
                log_rest_band=np.log10(3368/(1+z)) 

            else:    

                ## check this!!@!!!!!!!!!!!!!!!
                log_rest_band=np.log10(lambda_eff[bands.index(band)]/(1+z))
            M_ab_rest = np.array([np.interp(log_rest_band[i], 
                    log_lambda_eff, M_ab_bands[i, :]) 
                      for i in range(len(z))])

            m = M_ab_rest + 5 * np.log10(D_L)  -5
            print("m shape: ",np.shape(m))
            return m

        def target_filter_bgs(redshift):
            m_r=band_apparent_magnitude(z=redshift,band='r')
            mask = (m_r<=self.m_r_cutoff)
            return mask
        
        def target_filter_lrg(redshift):
            m_r=band_apparent_magnitude(z=redshift,band='r')
            m_z=band_apparent_magnitude(z=redshift,band='z')
            m_g=band_apparent_magnitude(z=redshift,band='g')
            m_w1=band_apparent_magnitude(z=redshift,band='w1')

            #here are all the selection criteria of the LRG survey
            mask= (m_z-m_w1>0.8*(m_r-m_z)-0.6)
            mask &= ((m_g-m_w1>2.9) | (m_r-m_w1>1.8))
            mask &= ((m_r-m_w1>1.8*(m_w1-17.4)) & 
                    ((m_r-m_w1>m_w1-16.33) | (m_r-m_w1>3.3)))
            print("mask shape: ",np.shape(mask))
            return mask


        if self.survey == 'bgs':
            m_mask=target_filter_bgs(redshift=coordinates[:,2])
        if self.survey == 'lrg':
            m_mask=target_filter_lrg(redshift=coordinates[:,2])
            
        pass_fraction = np.count_nonzero(m_mask) / len(m_mask) * 100
        print(f"Pass percentage after luminosity filter of \
galaxies with stellar mass: {pass_fraction:.2f}%")
        
        #updating internal total_mask
        old_mask[old_mask]=m_mask
        self.total_mask=old_mask
        
        coordinates=coordinates[m_mask]
        print("Luminosity filter applied")
        
        
        
        
        
        ##for testing of luminosity only!!
        #np.savetxt('redshift_data.txt',coordinates[:,2])
        return coordinates

