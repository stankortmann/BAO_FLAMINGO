from swiftsimio import load
import numpy as np
import psutil
import os
import time
import gc
import threading
import h5py
import unyt as u
import warnings
import copy




# Own modules
from baoflamingo.galaxy_correlation import correlation_tools
from baoflamingo.filtering import filtering_tools
from baoflamingo.cosmology import cosmo_tools
from baoflamingo.coordinates import coordinate_tools
from baoflamingo.template import template_CAMB



# Suppress only RuntimeWarnings from swiftsimio about cosmo_factors
#this has to be taken account of when changing coordinates from comoving to physical etc.

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"Mixing arguments with and without cosmo_factors.*"
)

# --- Memory and monitor ---
#every 240 seconds print CPU and memory usage
def monitor_system(interval=240,rank_id=0):
    """
    Print CPU and memory usage every `interval` seconds.
    - CPU usage: percentage of total CPU
    - Memory usage: resident set size (RSS) of the current process in GB
    """
    process = psutil.Process(os.getpid())
    while True:
        # CPU usage over the last interval
        cpu_percent = psutil.cpu_percent(interval=None)  # instantaneous
        # Per-process memory usage
        mem_gb = process.memory_info().rss / (1024 ** 3)
        
        print(f"[SYSTEM MONITOR] [Rank {rank_id}] CPU: {cpu_percent:.1f}% | Memory: {mem_gb:.3f} GB")
 
        time.sleep(interval)



def run_pipeline_single(cfg,mpi_comm,mpi_rank,mpi_size):
    """Main pipeline using the YAML config object."""
    
    # Start memory monitor
    if cfg.monitoring.cpu_ram_monitor:
        monitor_thread = threading.Thread(
        target=monitor_system,
        kwargs={"interval": cfg.monitoring.monitor_interval, "rank_id": mpi_rank},
        daemon=True
        )
        monitor_thread.start()
    
    # --- Paths ---
    directory = cfg.paths.directory
    simulation = cfg.paths.simulation
    soap_hbt_path = os.path.join(directory, simulation, cfg.paths.soap_hbt_subpath)
    redshift_path = os.path.join(directory, simulation, cfg.paths.redshift_file)
    
    # --- Load snapshot and redshift ---
    redshift_list = np.loadtxt(redshift_path, skiprows=1)
    redshift_number = cfg.paths.snapshot_number
    
    
    #all the data loading is only done by rank 0 and then broadcasted when
    #the correlation calculation starts
    if mpi_rank ==0:
        data = load(soap_hbt_path + str(int(redshift_number)) + ".hdf5")
        print("Snapshot", redshift_number, "loaded")
    
        # --- Cosmology ---
        metadata = data.metadata
        redshift = redshift_list[redshift_number]
        
        print("Snapshot redshift:", redshift)
        simulation_cosmology = metadata.cosmology
        box_size = metadata.boxsize[0]
        
        print(f"The size of the box is {box_size:.2f}")
        centre = np.array([box_size.value / 2] * 3)
        
        
        cosmo_real = cosmo_tools(
            box_size=box_size,
            constants=simulation_cosmology,
            redshift=redshift,
            redshift_bin_width=cfg.slicing.redshift_bin_width #delta_z
        )
        
        print("Cosmology set up")
        print("Centre radius is",cosmo_real.center_bin)
        print("Thickness of slice is:", cosmo_real.delta_dr)
    
    
    
        # --- Observer position ---
        if cosmo_real.complete_sphere:
            
            
            
            observer = centre.copy()
            
            print("Complete spherical slice is possible")
        
        #We have to take a look at this!
        
        else:
            
            
            #we have to do extra slicing with this one!
            observer = centre.copy()
            #x position so it is 'safe_offset' away from the 'wall' of the box
            #and in the middle of the y and z directions
            safe_offset= 1*u.Mpc #away from the box edge to avoid boundary issues
            observer[0] = cosmo_real.outer_edge_bin.value + safe_offset.value

            #testing if the partial slice is possible due to limitations of the box size
            #in the x direction. This can be proven geometrically.
            fail_safe =safe_offset+cosmo_real.delta_dr+\
            (1-np.cos(cosmo_real.max_angle))*cosmo_real.inner_edge_bin
            
            if fail_safe> box_size:
                
                print("No complete spherical slice is possible due to redshift bin width and max angle.")
                #it has to exit the entire pipeline
                exit()
            
            print("No complete spherical slice is possible")



        # --- Coordinate transformation set-up ---
        coordinates = coordinate_tools(
            cosmology=cosmo_real,
            observer=observer,
        )


        
        # --- Filtering set up---
        #we select the targets on their real cosmology
        #this eventually really dictates the target selection
        filters = filtering_tools(
            soap_file=data,
            cosmology=cosmo_real,
            cfg=cfg, #all the inputs of the configuration
            rank_id=mpi_rank
        )
        gc.collect()
        print("Filters set up")


        # --- Load in galaxy centers ---
        #might have to change this due to the fact that these are not actually
        #the centers of the galaxies but their respective haloes.
        d_coords = data.inclusive_sphere_50kpc.centre_of_mass.value

        # --- Stellar mass filter and nonzero luminosity ---
        # Apply and keep the filtered coordinates
        d_coords = filters.stellar_mass_filter(d_coords)
        d_coords = filters.zero_luminosity_filter(d_coords)

        # --- Convert to spherical coordinates and introduce error in redshift---
        d_coords_sph = coordinates.cartesian_to_spherical(d_coords) #(r,theta,phi)

        # Free memory
        del d_coords
        gc.collect()

        # --- Radial filter ---
        # This one will update total_mask and return (theta, phi,z)
        # Also introduces errors in the redshift data (DESI target selection maximum error)
        d_coords_sph = filters.radial_filter(d_coords_sph)

        # --- Redshift filter ---
        #this one will finally filter over the redshifts with error so the ones now outside
        #the redshift bin will also be filtered out. This will not be a lot
        d_coords_sph= filters.redshift_filter(d_coords_sph)

        # --- Luminosity filter ---
        # Applies apparent magnitude cut, also outputs (theta, phi,z) 
        d_coords_sph = filters.luminosity_filter(d_coords_sph)
        data_size = np.shape(d_coords_sph)[0]
        if data_size < 2:
            
            print("Empty slice after redshift filtering.")
        else:
            print("Data size after filtering:", data_size)

        # --- Fiducial cosmologies ---
        """
        Up until now we have filtered the galaxies according to the real cosmology of the 
        simulation box. Now we can proceed to the correlation function calculation. We can do this
        for either the real cosmology or a fiducial cosmology that is different from the real one.
        This is set in the config file.
        """
        param_dicts = []

        
        if cfg.fiducial.manual_cosmo:
            #number of parameters to vary
            n_params=len(cfg.fiducial.parameters_mcmc)

            if n_params == 1:
                p1_name = cfg.fiducial.parameters_mcmc[0]
                for p1 in np.linspace(*cfg.fiducial.para_1_range, cfg.fiducial.points_per_para):
                    param_dicts.append({
                        p1_name: p1,
                        "name": f"{p1_name}_{p1:.4f}"
                    })

            elif n_params == 2:
                p1_name, p2_name = cfg.fiducial.parameters_mcmc
                p1_vals = np.linspace(*cfg.fiducial.para_1_range, cfg.fiducial.points_per_para)
                p2_vals = np.linspace(*cfg.fiducial.para_2_range, cfg.fiducial.points_per_para)

                for p1 in p1_vals:
                    for p2 in p2_vals:
                        param_dicts.append({
                            p1_name: p1,
                            p2_name: p2,
                            "name": f"{p1_name}_{p1:.4f}_{p2_name}_{p2:.4f}"
                        })
                        

    # --- Broadcast filtered coordinates to all ranks ---
    if mpi_rank != 0:
        d_coords_sph = None
        cosmo_real = None
        box_size = None
        redshift = None
        simulation_cosmology = None
        param_dicts = None





    # --- Broadcast filtered coordinates to all ranks ---
    d_coords_sph = mpi_comm.bcast(d_coords_sph, root=0)
    cosmo_real = mpi_comm.bcast(cosmo_real, root=0)
    box_size = mpi_comm.bcast(box_size, root=0)
    redshift = mpi_comm.bcast(redshift, root=0)
    simulation_cosmology = mpi_comm.bcast(simulation_cosmology, root=0)
    param_dicts = mpi_comm.bcast(param_dicts, root=0)
    if mpi_rank ==0:
        print("Data broadcasted across all ranks.")


    


        
    

    #we will now use MPI parallelisation to distribute the different cosmologies across ranks
    
    #dividing up the cosmologies over the different ranks
   # divide among ranks
    local_param_dicts = param_dicts[mpi_rank::mpi_size]

    # each rank builds its OWN cosmology objects
    local_cosmo_list = []
    #add the real cosmology to the last rank, This one will always have less cosmologies to process
    if mpi_rank == mpi_size-1:
        local_cosmo_list.append(cosmo_real)
    for p in local_param_dicts:
        cosmo_new = cosmo_real.update(params=p)
        local_cosmo_list.append(cosmo_new)

    
    
    #       --- Correlation ---
    filenames={}
    #will be done for all cosmologies provided
    for cosmo in local_cosmo_list:

        # --- pycorr ---
        correlation = correlation_tools(
                coordinates=d_coords_sph,
                cosmology=cosmo,
                cfg=cfg,
                rank_id=mpi_rank
        
        )
        
        # --- CAMB template to fit later on ---
        if cfg.plotting.monopole or cfg.plotting.quadrupole:
            if cfg.plotting.monopole and cfg.plotting.quadrupole:
                l_list=[0,2]
            elif cfg.plotting.monopole and not cfg.plotting.quadrupole:
                l_list=[0]
            elif cfg.plotting.quadrupole and not cfg.plotting.monopole:
                l_list=[2]
            template = template_CAMB(cosmo=cosmo,
                                    effective_redshift=correlation.effective_redshift,
                                    s_array=correlation.s_bin_centers, 
                                    ell_list=l_list)

            # --- Get multipoles ---
            template_xi_dict = template.get_multipoles()
            template_xi0 = np.asarray(template_xi_dict[0],dtype=np.float64)  # monopole
            template_xi2 = np.asarray(template_xi_dict[2],dtype=np.float64)  # quadrupole (should be ~0 for ΛCDM)
        
        
        #print(f"Survey density in cosmology {cosmo.name}: {correlation.survey_density}")
        #print(f"Survey volume in cosmology {cosmo.name}: {correlation.survey_volume}")
        
        

        

        # --- Save output using HDF5 ---
        #realization number from SLURM array
        realization = int(os.environ.get("REALIZATION_ID", 0))
    
        # Construct nested output directory structure
        output_dir = os.path.join("results",cfg.paths.output_directory, simulation, str(redshift),f"run_{realization:03d}")

        # Make sure it exists (this creates all intermediate subdirectories)
        os.makedirs(output_dir, exist_ok=True)

        # Build the full file path
        output_filename = os.path.join(
            output_dir,
            f"{cosmo.name}.hdf5"
        )
        #save the filename to the dictionary
        filenames[cosmo.name]=output_filename

        


        with h5py.File(output_filename, "w") as f:
            f.create_dataset("name", data=cosmo.name)
            # --- Unitless arrays / scalars ---
            f.create_dataset("oversampling_factor", data=cfg.random_catalog.oversampling)
            f.create_dataset("random_size", data=correlation.n_random)
            f.create_dataset("delta_z", data=cfg.slicing.redshift_bin_width)
            f.create_dataset("effective_redshift", data=correlation.effective_redshift)
            
            #most important data!!!!
            #simulation data
            f.create_dataset("xi", data=correlation.xi)
            f.create_dataset("cov", data=correlation.cov)
            f.create_dataset("s_bin_centers", data=correlation.s_bin_centers)
            f["s_bin_centers"].attrs["units"] = str(u.Mpc)
            f.create_dataset("mu_bin_centers", data=correlation.mu_bin_centers)
            
            #CAMB data
            if cfg.plotting.monopole or cfg.plotting.quadrupole:
                f.create_dataset("template_s", data=template.s)
                f["template_s"].attrs["units"] = str(u.Mpc)
                if cfg.plotting.monopole:
                    f.create_dataset("template_xi0",data=template_xi0)   
                if cfg.plotting.quadrupole:
                    f.create_dataset("template_xi2",data=template_xi2)


            # --- Arrays / scalars with units ---
            f.create_dataset("slice_thickness", data=cosmo.delta_dr.value)
            f["slice_thickness"].attrs["units"] = str(cosmo.delta_dr.units)

            f.create_dataset("survey_density", data=correlation.survey_density.value)
            f["survey_density"].attrs["units"] = str(correlation.survey_density.units)
            
            f.create_dataset("survey_area", data=correlation.survey_area.value)
            f["survey_area"].attrs["units"] = str(correlation.survey_area.units)

            f.create_dataset("survey_volume", data=correlation.survey_volume.value)
            f["survey_volume"].attrs["units"] = str(correlation.survey_volume.units)
            
            f.create_dataset("survey_total_volume_percent", data=correlation.survey_total_volume_percent)
            
            #save the bao distance of the fiducial cosmology used and other cosmo quantities
            f.create_dataset("BAO_distance", data=cosmo.bao_distance.value)
            f["BAO_distance"].attrs["units"] = str(cosmo.bao_distance.units)


            f.create_dataset("effective_radius", data=correlation.effective_radius.value)
            f["effective_radius"].attrs["units"] = str(correlation.effective_radius.units)

            f.create_dataset("effective_H_z", data=correlation.effective_H_z.value)
            f["effective_H_z"].attrs["units"] = str(correlation.effective_H_z.units)

            f.create_dataset("effective_D_a", data=correlation.effective_D_a.value)
            f["effective_D_a"].attrs["units"] = str(correlation.effective_D_a.units)


            #attributes to save from cfg.fiducial
            for attr in cfg.fiducial.parameters_to_save:
                if hasattr(cosmo, attr):
                    f.create_dataset(attr, data=getattr(cosmo, attr))
                    

            

            

            
            
        
    
    all_filenames_dicts = mpi_comm.gather(filenames, root=0)

    if mpi_rank == 0:
        # Combine dictionaries from all ranks
        combined_filenames = {}
        for d in all_filenames_dicts:
            combined_filenames.update(d)
        
        return combined_filenames
    else:
        return None