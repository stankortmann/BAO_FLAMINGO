from dataclasses import dataclass
from typing import List, Union

# --- PATH STRUCTURE ---
@dataclass
class Paths:
    base: str
    results_dir: str
    simulation: str
    redshift: Union[float, List[float]]
    cosmology: Union[int, List[int]]


# --- PLOTTING STRUCTURE ---
@dataclass
class Plotting:
    correlation_2d: bool
    variance_2d: bool
    correlation_1d: bool
    expected_bao_position: float   # in Mpc
    bao_window: float              # in Mpc
    plot_bao: bool
    mu_rebin: int
    s_rebin: int


# --- ROOT CONFIG OBJECT ---
@dataclass
class Config:
    paths: Paths
    plotting: Plotting
    