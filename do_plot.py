import yaml
import numpy as np
import argparse
import os
import re
import matplotlib.pyplot as plt


# Our own modules
import baoflamingo.data_structure as ds
from baoflamingo.pipeline_single import run_pipeline_single
from baoflamingo.plotting import correlation_plotter
from baoflamingo.plotting import posterior_plotter 



if __name__ == "__main__":

      # --- Argument parser ---
    parser = argparse.ArgumentParser(description="Run BAOflamingo PyCorr pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/mcmc_2d.yaml",
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



    data_directory = cfg.paths.data_directory
    simulation = cfg.paths.simulation
    output_directory=cfg.paths.output_directory
    results_directory=cfg.paths.results_directory
    redshift_path = os.path.join(data_directory, simulation, cfg.paths.redshift_file)
    redshift_list = np.loadtxt(redshift_path, skiprows=1)
    # Build base directory where results are stored
    BASE_RESULTS = os.path.join(output_directory,"results",results_directory,simulation)
     # --- Handle multiple snapshot numbers ---
    snapshot_numbers = cfg.paths.snapshot_number

    #relative error, hardcoded for now:
    relative_error=0.05
    
    if isinstance(snapshot_numbers, int):
        snapshot_numbers = [snapshot_numbers]

    for snap in snapshot_numbers:
        redshift = redshift_list[snap]
        redshift_dir = os.path.join(BASE_RESULTS, str(redshift))

        if not os.path.isdir(redshift_dir):
            print(f"Redshift folder missing: {redshift_dir}")
            continue

        print(f"\n--- Replotting redshift {redshift} ---")

        # Find all run directories like run_000, run_001, ...
        run_dirs = []
        for d in os.listdir(redshift_dir):
            if re.match(r"run_\d{3}", d) and os.path.isdir(os.path.join(redshift_dir, d)):
                run_dirs.append(d)
        run_dirs.sort()  # optional: process in order

        for run_dir in run_dirs:
            full_run_path = os.path.join(redshift_dir, run_dir)
            # Find all .hdf5 files in this run directory
            mcmc_list=[]
            files = [f for f in os.listdir(full_run_path) if f.endswith(".hdf5")]
            for fname in files:
                data_filename = os.path.join(full_run_path, fname)
                print(f"Reading {data_filename}")
                
                plotter = correlation_plotter(
                    filename=data_filename,
                    cfg=cfg,
                    mu_rebin=1,
                    s_rebin=1
                )

                #entering the values of the template with shift method
                if cfg.fiducial.manual_cosmo and hasattr(plotter, "alpha_with_shift"):
                    name=plotter.name.decode()
                    n_parameters=len(cfg.fiducial.parameters_mcmc)
                    
                    if n_parameters==1 :
                        cosmo_real_para_value=None
                        if name != "w0waCDM":
                       
                            entry = {
                            "name": name,
                            "para_name": plotter.para_name,
                            "para_value": float(plotter.para_value),
                            "alpha_mean": plotter.alpha_with_shift[0],
                            "alpha_std": plotter.alpha_with_shift[1],
                            "quad_mean": plotter.mu_quad,
                            "quad_std": plotter.std_quad
                            } 
                            if entry["alpha_mean"] is not None and entry["alpha_std"]/entry["alpha_mean"] < relative_error:
                                mcmc_list.append(entry)
                        if name == "w0waCDM":
                            true_pars={"para":float(plotter.para1_value)}
                            
                    elif n_parameters==2 :
                        cosmo_real_para1_value=None
                        cosmo_real_para2_value=None
                        #for two parameters, para1 and para2 are stored
                        if name != "w0waCDM":
                            entry = {
                            "name": name,
                            "para1_name": plotter.para1_name,
                            "para2_name": plotter.para2_name,
                            "para1_value": float(plotter.para1_value),
                            "para2_value": float(plotter.para2_value),
                            "alpha_mean": plotter.alpha_with_shift[0],
                            "alpha_std": plotter.alpha_with_shift[1],
                            "quad_mean": plotter.mu_quad,
                            "quad_std": plotter.std_quad
                            } 
                            if entry["alpha_mean"] is not None and entry["alpha_std"]/entry["alpha_mean"] < relative_error:
                                mcmc_list.append(entry)

                        if name == "w0waCDM":
                            true_pars={"para1":float(plotter.para1_value), \
                                        "para2":float(plotter.para2_value)}

            if cfg.fiducial.manual_cosmo:
                posterior_plotter( 
                redshift=redshift,
                mcmc_list=mcmc_list, 
                outdir=full_run_path,
                use_quad_likelihood=True,
                true_pars=true_pars,
                provided_likelihoods=None
                )
                
        print("\nAll snapshots processed successfully.")


