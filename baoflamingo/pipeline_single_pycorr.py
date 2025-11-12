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

# Our own modules
from baoflamingo.galaxy_correlation_pycorr import correlation_tools_pycorr
from baoflamingo.filtering import filtering_tools
from baoflamingo.statistics import stat_tools
from baoflamingo.cosmology import cosmo_tools
from baoflamingo.coordinates import coordinate_tools



# Suppress only RuntimeWarnings from swiftsimio about cosmo_factors
#this has to be taken account of when changing coordinates from comoving to physical etc.

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"Mixing arguments with and without cosmo_factors.*"
)

# --- Memory and monitor ---

def monitor_system(interval=120):
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
        
        print(f"[SYSTEM MONITOR] CPU: {cpu_percent:.1f}% | Memory: {mem_gb:.3f} GB")
 
        time.sleep(interval)



def run_pipeline_single(cfg):
    """Main pipeline using the YAML config object."""
    
    # Start memory monitor
    monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    monitor_thread.start()
    
    # --- Paths ---
    directory = cfg.paths.directory
    simulation = cfg.paths.simulation
    soap_hbt_path = os.path.join(directory, simulation, cfg.paths.soap_hbt_subpath)
    redshift_path = os.path.join(directory, simulation, cfg.paths.redshift_file)
    
    # --- Load snapshot and redshift ---
    redshift_list = np.loadtxt(redshift_path, skiprows=1)
    redshift_number = cfg.paths.snapshot_number
    
    
    data = load(soap_hbt_path + str(int(redshift_number)) + ".hdf5")
    print("Snapshot", redshift_number, "loaded")
    
    # --- Cosmology ---
    metadata = data.metadata
    redshift = redshift_list[redshift_number]
    print("Snapshot redshift:", redshift)
    simulation_cosmology = metadata.cosmology
    box_size = metadata.boxsize[0]
    print(box_size)
    centre = np.array([box_size.value / 2] * 3)
    
    cosmo_real = cosmo_tools(
        box_size=box_size,
        constants=simulation_cosmology,
        redshift=redshift,
        redshift_bin_width=cfg.slicing.redshift_bin_width #delta_z
    )
    print("Cosmology set up")
    print("Centre radius is",cosmo_real.center_bin)
    del metadata
    gc.collect()
    
    print("Thickness of slice is:", cosmo_real.delta_dr)
    
    # --- Observer position ---
    if cosmo_real.complete_sphere:
        
        shift = np.random.uniform(-0.5*box_size.value, 0.5*box_size.value, size=3)
        
        observer = centre.copy()
        
        print("Complete spherical slice is possible")
    
    #We have to take a look at this!
    
    else:
        
        shift = np.random.uniform(-0.5*box_size.value, 0.5*box_size.value, size=3)
        #we have to do extra slicing with this one!
        observer = centre.copy()
        #shift of the x value!
        observer[0] += cosmo_real.center_bin.value
        print("No complete spherical slice is possible")



    # --- Coordinate transformation set up ---
    coordinates = coordinate_tools(
        cosmology=cosmo_real,
        observer=observer,
        shift=shift
    )


    
    # --- Filtering set up---
    #we select the targets on their real cosmology
    #this eventually really dictates the target selection
    filters = filtering_tools(
        soap_file=data,
        cosmology=cosmo_real,
        cfg=cfg #all the inputs of the configuration
    )
    gc.collect()
    print("Filters set up")


    # --- Load in galaxy centers ---
    #might have to change this due to the fact that these are not actually
    #the centers of the galaxies but their respective haloes.
    d_coords = data.input_halos.halo_centre.value

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
        return
    
    print("Data size after filtering:", data_size)


    """
    Up until now we have filtered the galaxies according to the real cosmology of the 
    simulation box. Now we can proceed to the correlation function calculation. We can do this
    for either the real cosmology or a fiducial cosmology that is different from the real one.
    This is set in the config file.
    """
    from colossus.cosmology import cosmology as cosmo_fiducial
    #setting the cosmology to the one provided in the config file
    cosmo_fiducial.setCosmology(cfg.fiducial.cosmology)

    cosmo_fid = cosmo_tools(
        box_size=box_size,
        constants=cosmo_fiducial.current_cosmo,
        redshift=redshift,
        redshift_bin_width=cfg.slicing.redshift_bin_width #delta_z
        )


    #might add more cosmologies later on, so we do a list
    cosmo_list = [cosmo_real,cosmo_fid]
    filenames=[]


    # --- Correlation ---

    #will be done for all cosmologies provided
    for cosmo in cosmo_list:
        correlation = correlation_tools_pycorr(
                coordinates=d_coords_sph,
                cosmology=cosmo,
                cfg=cfg
        
        )
        
        

        print("Survey density :", correlation.survey_density)
        
        



        # --- Save output using HDF5 ---

        # Construct nested output directory structure
        output_dir = os.path.join(cfg.paths.output_directory, simulation, str(redshift))

        # Make sure it exists (this creates all intermediate subdirectories)
        os.makedirs(output_dir, exist_ok=True)

        # Build the full file path
        output_filename = os.path.join(
            output_dir,
            f"{cosmo.name}.hdf5"
        )

        print("Saving results to:", output_filename)


        with h5py.File(output_filename, "w") as f:

            # --- Unitless arrays / scalars ---
            f.create_dataset("oversampling_factor", data=cfg.random_catalog.oversampling)
            f.create_dataset("random_size", data=correlation.n_random)
            f.create_dataset("ls_avg", data=correlation.ls_avg)
            f.create_dataset("ls_std", data=correlation.ls_std)

            # --- Arrays / scalars with units ---
            f.create_dataset("slice_thickness", data=cosmo_real.delta_dr.value)
            f["slice_thickness"].attrs["units"] = str(cosmo_real.delta_dr.units)

            f.create_dataset("survey_density", data=correlation.survey_density.value)
            f["survey_density"].attrs["units"] = str(correlation.survey_density.units)
            
            f.create_dataset("survey_area", data=correlation.survey_area.value)
            f["survey_density"].attrs["units"] = str(correlation.survey_area.units)


            f.create_dataset("s_bin_centers", data=correlation.s_bin_centers)
            f["s_bin_centers"].attrs["units"] = str(u.Mpc)

            f.create_dataset("mu_bin_centers", data=correlation.mu_bin_centers)
            


        print("Saved single-slice Landy-Szalay histogram to", output_filename)
        filenames.append(output_filename)
    return filenames #for plotting in the same pipeline!