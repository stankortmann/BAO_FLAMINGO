from dataclasses import dataclass
from typing import List, Union

@dataclass
class Paths:
    directory: str
    soap_hbt_subpath: str
    simulation: str
    redshift_file: str
    snapshot_number: Union[int, List[int]]  # can be a single int or list
    output_directory:str

@dataclass
class Slicing:
    redshift_bin_width: float
    

@dataclass
class RandomCatalog:
    seed: int
    oversampling: int

@dataclass
class Filters:
    central_filter: bool
    stellar_mass_filter: bool
    stellar_mass_cutoff: float
    luminosity_filter: bool
    survey: str
    m_r_cutoff: float
    
@dataclass
class Plotting:
    mu_bins: int
    mu_min: float
    mu_max: float
    
    s_bins: int
    s_min: float
    s_max: float

@dataclass
class Statistics:
    n_patches: Union[int, List[int]]

@dataclass
class Config:
    paths: Paths
    slicing: Slicing
    random_catalog: RandomCatalog
    filters: Filters
    plotting: Plotting
    statistics: Statistics
