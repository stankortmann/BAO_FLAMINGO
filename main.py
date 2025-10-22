
import yaml

# Our own modules
import baoflamingo.data_structure as ds  # dataclasses in separate file
from baoflamingo.pipeline_single import run_pipeline_single
from baoflamingo.plotting import plot_correlation_single_slice


############-------ACTUAL RUNNING, DO NOT DELETE!!! -------#############

if __name__ == "__main__":
    # --- Load YAML config file to your liking ---
    with open("configurations/config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    #ds. =data_structure.py, add inputs here!
    cfg = ds.Config(
        paths=ds.Paths(**cfg_dict['paths']),
        slicing=ds.Slicing(**cfg_dict['slicing']),
        distance=ds.Distance(**cfg_dict['distance']),
        random_catalog=ds.RandomCatalog(**cfg_dict['random_catalog']),
        filters=ds.Filters(**cfg_dict['filters']),
        plotting=ds.Plotting(**cfg_dict['plotting']),
        statistics=ds.Statistics(**cfg_dict['statistics'])
    )
    if cfg.slicing.method=='single':
        data_filename=run_pipeline_single(cfg)
        data_plot=plot_correlation_single_slice(
            cfg=cfg,
            filename=data_filename,
            show_fit=True, #this is the polynomial fit, not the spline
            save_plot=True)
    
    #Will be implemented later on!
    """ 
    if cfg.slicing.method=='multiple':
        run_pipeline_multiple(cfg)

    """


