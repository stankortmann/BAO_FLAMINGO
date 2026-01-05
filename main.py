import yaml
import numpy as np
import argparse
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD  # global communicator for all processes
rank = comm.Get_rank()  # integer ID of this process, 0..(size-1)
size = comm.Get_size()  # total number of processes
if rank ==0:
    print(f"The ammount of processes running: {size}")

# Our own modules
import baoflamingo.data_structure as ds
from baoflamingo.pipeline_single import run_pipeline_single
from baoflamingo.plotting import correlation_plotter 


############-------ACTUAL RUNNING, DO NOT DELETE!!! -------#############

if __name__ == "__main__":

      # --- Argument parser ---
    parser = argparse.ArgumentParser(description="Run BAOflamingo PyCorr pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/config_pycorr.yaml",
        help="Path to the YAML configuration file"
    )
    args=parser.parse_args()
    # --- Load YAML config file ---
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Create config object
    cfg = ds.Config(
        paths=ds.Paths(**cfg_dict['paths']),
        monitoring=ds.Monitoring(**cfg_dict['monitoring']),
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
    if rank==0:
        print(f"Found snapshot numbers: {snapshot_numbers}", flush=True)
        print(f"This is simulation: {cfg.paths.simulation}", flush=True)

    # --- Run pipeline for each snapshot ---
    for snap in snapshot_numbers:
        if rank==0:
            print(f"\n=== Running for snapshot {snap} ===", flush=True)
        cfg.paths.snapshot_number = snap  # update for this iteration

        # --- Running the pipeline ---
        data_filenames = run_pipeline_single(cfg,
            mpi_comm=comm,mpi_rank=rank,mpi_size=size)
        

    comm.Barrier()            
    if rank==0:                
        print("\nAll snapshots processed successfully.")


