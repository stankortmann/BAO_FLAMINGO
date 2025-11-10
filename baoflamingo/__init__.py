# BAO/baoflamingo/__init__.py
from . import coordinates
from . import cosmology
from . import data_structure
from . import data_structure_pycorr
from . import filtering
from . import galaxy_correlation
from . import galaxy_correlation_pycorr
#from . import multiple_slices
#from . import old_correlation
from . import pipeline_single
from . import pipeline_single_pycorr
from . import plotting
from . import plotting_pycorr
#from . import single_slice
#from . import smooth_fitting
#from . import statistics

__all__ = [
    "coordinates",
    "cosmology",
    "data_structure",
    "data_structure_pycorr",
    "filtering",
    "galaxy_correlation",
    "galaxy_correlation_pycorr",
    #"multiple_slices",
    #"old_correlation",
    "pipeline_single",
    "pipeline_single_pycorr",
    "plotting",
    "plotting_pycorr",
    #"single_slice",
    #"smooth_fitting",
    #"statistics",
]

