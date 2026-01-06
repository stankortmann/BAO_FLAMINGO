# Standard modules
import yaml
import numpy as np
import argparse
import os
import re
import matplotlib.pyplot as plt


# Our own modules
import baoflamingo.data_structure as ds

from baoflamingo.plotting import correlation_plotter
from baoflamingo.plotting import posterior_plotter 
from baoflamingo.averaging import xi_cov_averager



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
    base = os.path.join(output_directory,"results",results_directory,simulation)
    # Build directory for the combined posterior results
    combined_posterior_dir = os.path.join(base, "combined_posteriors")
    os.makedirs(combined_posterior_dir, exist_ok=True)
    combined_likelihoods = None #multiply all the likelihoods together
     # --- Handle multiple snapshot numbers ---
    snapshot_numbers = cfg.paths.snapshot_number

    
    if isinstance(snapshot_numbers, int):
        snapshot_numbers = [snapshot_numbers]

    for snap in snapshot_numbers:

        redshift = redshift_list[snap]
        redshift_dir = os.path.join(base, str(redshift))

        if not os.path.isdir(redshift_dir):
            print(f"Redshift folder missing: {redshift_dir}")
            continue

        #averaging over all the runs and writing the average output in a new folder 'average' inside redshift_dir
        print(f"\n--- Averaging correlation functions at redshift {redshift} ---")
        averager = xi_cov_averager(base_dir=redshift_dir)
        

        #unpack the average run directory
        average_run_path = os.path.join(redshift_dir,"average")
        #the rest is the same as in do_plot.py, but now for the average run!
        mcmc_list=[]

        relative_error=0.1  #maximum relative error to include the point in the posterior plot
        files = [f for f in os.listdir(average_run_path) if f.endswith(".hdf5")]
        for fname in files:
            data_filename = os.path.join(average_run_path, fname)
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
                        "quad_std": plotter.std_quad,
                        "fit_status": True
                        } 
                        if entry["alpha_mean"] is not None and entry["alpha_std"]/entry["alpha_mean"] < relative_error:
                            mcmc_list.append(entry)
                        else:
                            entry["fit_status"] = False
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
            posterior_init=posterior_plotter( 
                    cfg=cfg,
                    redshift=redshift,
                    mcmc_list=mcmc_list, 
                    outdir=average_run_path,
                    use_quad_likelihood=True,
                    true_pars=true_pars,
                    provided_likelihoods=None
                    )
            #store the combined likelihoods over all snapshots by multiplying them
            #this assumes that the snapshots are independent which is not strictly true
            if combined_likelihoods is None:
                combined_likelihoods = posterior_init.likelihood
            else:
                combined_likelihoods *= posterior_init.likelihood
    print("\nAll snapshots processed successfully.")       
    if cfg.fiducial.manual_cosmo and combined_likelihoods is not None:
        posterior_plotter( 
                cfg=cfg,
                redshift=None,
                mcmc_list=mcmc_list, #use the most recent mcmc_list for the parameter names
                outdir=combined_posterior_dir,
                use_quad_likelihood=True,
                true_pars=true_pars, #use the most recent true_pars
                provided_likelihoods=combined_likelihoods
                )   
        print("\nCombined posterior over all snapshots processed successfully.") 
    
