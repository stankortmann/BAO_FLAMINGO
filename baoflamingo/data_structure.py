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
    method:str
    n_slices:int
    n_sigma: float
    

@dataclass
class Distance:
    min: float
    max: float
    type: str

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
    band: str
    m_cutoff: float
    
@dataclass
class Plotting:
    bins: int
    leafsize: int

@dataclass
class Statistics:
    variance_method: str
    n_patches: int

@dataclass
class Config:
    paths: Paths
    slicing: Slicing
    distance: Distance
    random_catalog: RandomCatalog
    filters: Filters
    plotting: Plotting
    statistics: Statistics
