import numpy as np


class FilteringTools:
    def __init__(self,min_radius,max_radius,stellar_mass,
                gama_bands,redshift,central_mask=None):
        self.min_radius=min_radius
        self.max_radius=max_radius
        self.central_mask=central_mask
        self.gama_bands=gama_bands  
        self.redshift=redshift
        self.stellar_mass=stellar_mass





    def radial_filter(self,coordinates):
        """
        Filter coordinates within a spherical shell defined by min and max radius.

        Parameters
        ----------
        coordinates : np.ndarray, shape (N, 3)
            Spherical coordinates (r, theta, phi).
        min_radius : float
            Minimum radius.
        max_radius : float
            Maximum radius.

        Returns
        -------
        filtered_coords : np.ndarray
            Coordinates within the specified radial range.
        """
        r = coordinates[:, 0]
        mask = (r >= self.min_radius) & (r <= self.max_radius)
        return coordinates[mask]

    def central_filter(self,coordinates):
        
        return coordinates[self.central_mask]

    
    def stellar_mass_filter(self,coordinates)
        return 


    def luminosity_filter(self,coordinates):
        return
