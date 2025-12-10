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
class Monitoring:
    cpu_ram_monitor: bool
    monitor_interval: int

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

    expected_bao_position: float
    bao_window: float
    plot_bao: bool
    
    correlation_2d: bool
    autocovariance_2d: bool
    covariance_regularization: Union[float, None]

    monopole: bool
    fit_gaussian: bool
    fit_noshift: bool
    fit_shift: bool
    include_nuissance: bool
    nuissance_poly_order: int

    quadrupole: bool

@dataclass
class Statistics:
    n_patches: Union[int, List[int]]


@dataclass
class Fiducial:
    cosmology: Union[str,list[str]]
    manual_cosmo : bool # enable manual cosmology grid, if off the cosmologies are selected in the above cosmologies

    parameters_to_save: list[str]
    
    parameters_mcmc: list[str] 
    para_1_range: list[float]
    para_2_range: list[float]
    points_per_para: int



#important class that orders all the configurations
@dataclass
class Config:
    paths: Paths
    monitoring: Monitoring
    slicing: Slicing
    random_catalog: RandomCatalog
    filters: Filters
    plotting: Plotting
    statistics: Statistics
    fiducial: Fiducial
