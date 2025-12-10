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



    directory = cfg.paths.directory
    simulation = cfg.paths.simulation
    output_directory=cfg.paths.output_directory
    redshift_path = os.path.join(directory, simulation, cfg.paths.redshift_file)
    redshift_list = np.loadtxt(redshift_path, skiprows=1)
    # Build base directory where results are stored
    BASE_RESULTS = os.path.join("results",output_directory,simulation)
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
                if cfg.fiducial.manual_cosmoand hasattr(plotter, "alpha_with_shift"):
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
                            cosmo_real_para_value=float(plotter.para_value)
                            
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
                            cosmo_real_para1_value=float(plotter.para1_value)
                            cosmo_real_para2_value=float(plotter.para2_value)

            if cfg.fiducial.manual_cosmo:
                #all files in the redshift/runxxx are added, so now MCMC
               
                if n_parameters==1:
                    

                    # ---- Alpha plot ----
                    para = np.array([d["value_para"] for d in mcmc_list])
                    alpha = np.array([d["alpha_mean"] for d in mcmc_list])
                    alpha_err = np.array([d["alpha_std"] for d in mcmc_list])
                    plt.figure(figsize=(8,6))
                    plt.errorbar(
                                x=para, 
                                y=alpha,
                                yerr=alpha_err, 
                                fmt='o', capsize=3, 
                                label='Alpha with shift'
                                )
                    if real_cosmo_para_value is not None:
                        plt.axvline(cosmo_real_para_value, color="black", 
                        linestyle="--", linewidth=0.2,
                        label="Real cosmology")
                    variable=mcmc_list[0]["para_name"]
                    plt.xlabel(variable)  # parameter name, e.g. "Om0"
                    plt.ylabel(r'$\alpha$')
                    plt.title(f'Redshift {redshift:.2f}')  # replace redshift with your value
                    plt.grid(True)
                    plt.legend()
                    # --- Save figure ---
                    filename_mcmc=os.path.join(full_run_path,"alpha_mcmc.png")
                    plt.savefig(filename_mcmc, 
                    dpi=300)  # PNG file
                    
                    plt.close()
                    


                    # ---- Average Quadrupole plot ----
                    para = np.array([d["value_para"] for d in mcmc_list])
                    quad_mean = np.array([d["quad_mean"] for d in mcmc_list])
                    quad_std = np.array([d["quad_std"] for d in mcmc_list])
                    plt.figure(figsize=(8,6))
                    plt.errorbar(
                                x=para, 
                                y=quad_mean,
                                yerr=quad_std, 
                                fmt='o', capsize=3, 
                                label='quadrupole mean'
                                )
                    if real_cosmo_para_value is not None:
                        plt.axvline(cosmo_real_para_value, color="black", 
                        linestyle="--", linewidth=0.2,
                        label="Real cosmology")
                    variable=mcmc_list_sorted[0]["para_name"]
                    plt.xlabel(variable)  # parameter name, e.g. "Om0"
                    plt.ylabel(r'Average Quadrupole ξ₂')
                    plt.title(f'Redshift {redshift:.2f}')  # replace redshift with your value
                    plt.grid(True)
                    plt.legend()
                    # --- Save figure ---
                    filename_mcmc=os.path.join(full_run_path,"avg_quad.png")
                    plt.savefig(filename_mcmc, 
                    dpi=300)  # PNG file
                    
                    plt.close()


                    print(f"For redshift {redshift} the MCMC plot of the variable {variable} is plotted.")
                    
                if n_parameters==2:
                    # ---- Alpha contour plot ----
                    para1 = np.array([d["para1_value"] for d in mcmc_list])
                    para2 = np.array([d["para2_value"] for d in mcmc_list])
                    alpha = np.array([d["alpha_mean"] for d in mcmc_list])
                    alpha_err = np.array([d["alpha_std"] for d in mcmc_list])
                    
                    plt.figure(figsize=(8,6))
                    sc = plt.scatter(
                                x=para1, 
                                y=para2, 
                                c=alpha-1,
                                vmin=-0.1,
                                vmax=0.1, 
                                s=100, 
                                cmap='seismic', 
                                label='Alpha with shift'
                                )
                    plt.colorbar(sc, label=r'$\alpha-1$')
                    if cosmo_real_para1_value is not None:
                        plt.axvline(cosmo_real_para1_value, color="black",
                                     linestyle="--", linewidth=0.5)

                    if cosmo_real_para2_value is not None:
                        plt.axhline(cosmo_real_para2_value, color="black", 
                                        linestyle="--", linewidth=0.5)

                    plt.scatter(cosmo_real_para1_value, cosmo_real_para2_value, 
                                color="black", s=150, marker="x", label="Real cosmology")

                    variable1=mcmc_list[0]["para1_name"]
                    variable2=mcmc_list[0]["para2_name"]
                    plt.xlabel(variable1)  # parameter name, e.g. "Om0"
                    plt.ylabel(variable2)  # parameter name, e.g. "Ode0"
                    plt.title(f'Redshift {redshift:.2f}')  # replace redshift with your value
                    plt.grid(True)
                    # --- Save figure ---
                    filename_mcmc=os.path.join(full_run_path,"alpha_mcmc_contour.png")
                    plt.savefig(filename_mcmc, 
                    dpi=300)  # PNG file
                    
                    plt.close()
                    
                    print(f"For redshift {redshift} the MCMC contour plot of the variables {variable1} and {variable2} is plotted.")

                    # ---- Average Quadrupole contour plot ----
                    para1 = np.array([d["para1_value"] for d in mcmc_list])
                    para2 = np.array([d["para2_value"] for d in mcmc_list])
                    quad_mean = np.array([d["quad_mean"] for d in mcmc_list])
                    quad_std = np.array([d["quad_std"] for d in mcmc_list]) 

                    plt.figure(figsize=(8,6))
                    sc = plt.scatter(
                                x=para1, 
                                y=para2, 
                                c=quad_mean, 
                                s=100, 
                                vmin=np.min(quad_mean),
                                vmax=np.mean(quad_mean),
                                cmap='Reds', 
                                label='Average Quadrupole'
                                )
                    plt.colorbar(sc, label='Average Quadrupole ξ₂')
                    if cosmo_real_para1_value is not None:
                        plt.axvline(cosmo_real_para1_value, color="black", 
                                linestyle="--", linewidth=0.5)

                    if cosmo_real_para2_value is not None:
                        plt.axhline(cosmo_real_para2_value, color="black", 
                                linestyle="--", linewidth=0.5)

                    plt.scatter(cosmo_real_para1_value, cosmo_real_para2_value, 
                                color="black", s=150, marker="x", label="Real cosmology")
                    variable1=mcmc_list[0]["para1_name"]
                    variable2=mcmc_list[0]["para2_name"]
                    plt.xlabel(variable1)  # parameter name, e.g. "Om0"
                    plt.ylabel(variable2)  # parameter name, e.g. "Ode0"
                    plt.title(f'Redshift {redshift:.2f}')  # replace redshift with your value
                    plt.grid(True)
                    # --- Save figure ---
                    filename_mcmc=os.path.join(full_run_path,"avg_quad_contour.png")
                    plt.savefig(filename_mcmc, 
                    dpi=300)  # PNG file    
                    plt.close()
                    
                
        print("\nAll snapshots processed successfully.")


