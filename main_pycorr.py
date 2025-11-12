import yaml
import numpy as np

# Our own modules
import baoflamingo.data_structure_pycorr as ds
from baoflamingo.pipeline_single_pycorr import run_pipeline_single
from baoflamingo.plotting_pycorr_new import correlation_plotter 


############-------ACTUAL RUNNING, DO NOT DELETE!!! -------#############

if __name__ == "__main__":
    # --- Load YAML config file ---
    with open("configurations/config_pycorr.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Create config object
    cfg = ds.Config(
        paths=ds.Paths(**cfg_dict['paths']),
        slicing=ds.Slicing(**cfg_dict['slicing']),
        random_catalog=ds.RandomCatalog(**cfg_dict['random_catalog']),
        filters=ds.Filters(**cfg_dict['filters']),
        plotting=ds.Plotting(**cfg_dict['plotting']),
        statistics=ds.Statistics(**cfg_dict['statistics']),
        fiducial=ds.Fiducial(**cfg_dict['fiducial'])
    )

    # --- Handle multiple snapshot numbers ---
    snapshot_numbers = cfg.paths.snapshot_number
    # Ensure it's always iterable
    if isinstance(snapshot_numbers, int):
        snapshot_numbers = [snapshot_numbers]

    print(f"Found snapshot numbers: {snapshot_numbers}", flush=True)

    # --- Run pipeline for each snapshot ---
    for snap in snapshot_numbers:
        print(f"\n=== Running for snapshot {snap} ===", flush=True)
        cfg.paths.snapshot_number = snap  # update for this iteration

        
        data_filenames = run_pipeline_single(cfg)


        for data_filename in data_filenames:
            print(f"Generating plots for {data_filename}...", flush=True)
            plotter = correlation_plotter(
                filename=data_filename,
                cfg=cfg,
                mu_rebin=1, #subject to future change
                s_rebin=1 #subject to future change 
            )
        

        # Placeholder for future multiple-slice functionality
        # if cfg.slicing.method == 'multiple':
        #     run_pipeline_multiple(cfg)

    print("\nAll snapshots processed successfully.")


