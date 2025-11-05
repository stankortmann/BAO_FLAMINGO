import yaml
import numpy as np

# Our own modules
import baoflamingo.data_structure_pycorr as ds
from baoflamingo.pipeline_single_pycorr import run_pipeline_single
from baoflamingo.plotting_pycorr import plot_correlation_2d


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
        statistics=ds.Statistics(**cfg_dict['statistics'])
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

        
        data_filename = run_pipeline_single(cfg)
        data_plot = plot_correlation_2d(
            cfg=cfg,
            filename=data_filename,
            save_plot=True
        )

        # Placeholder for future multiple-slice functionality
        # if cfg.slicing.method == 'multiple':
        #     run_pipeline_multiple(cfg)

    print("\nAll snapshots processed successfully.")


