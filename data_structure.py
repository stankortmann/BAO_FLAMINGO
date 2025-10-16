from dataclasses import dataclass

@dataclass
class Paths:
    directory: str
    simulation: str
    soap_hbt_subpath: str
    redshift_file: str
    snapshot_number:int

@dataclass
class Slicing:
    method:str
    n_slices:int
    

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
    n_patches= int

@dataclass
class Config:
    paths: Paths
    snapshot: Snapshot
    distance: Distance
    random_catalog: RandomCatalog
    filters: Filters
    plotting: Plotting
    statistics: Statistics
