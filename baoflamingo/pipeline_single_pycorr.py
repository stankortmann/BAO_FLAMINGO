from swiftsimio import load
import numpy as np
import psutil
import os
import time
import gc
import threading
import h5py
import unyt as u

# Our own modules
import baoflamingo.galaxy_correlation_pycorr as gal_cor
import baoflamingo.filtering as flt
import baoflamingo.statistics as stat
import baoflamingo.cosmology as cs
from baoflamingo.coordinates import coordinate_tools

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
    cosmology = metadata.cosmology
    box_size = metadata.boxsize[0]
    print(box_size)
    centre = np.array([box_size.value / 2] * 3)
    
    cosmo_real = cs.cosmo_tools(
        box_size=box_size,
        H0=cosmology.H0.value,
        Omega_m=cosmology.Om0,
        Omega_b=cosmology.Ob0,
        Omega_lambda=cosmology.Ode0,
        Tcmb=cosmology.Tcmb0.value,
        Neff=cosmology.Neff,
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
    filters = flt.filtering_tools(
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
    d_coords_sph = coordinates.cartesian_to_spherical(d_coords) #(z,theta,phi)

    # Free memory
    del d_coords
    gc.collect()

    # --- Radial filter ---
    # This one will update total_mask and return (z,theta, phi) with 
    d_coords_sph = filters.radial_filter(d_coords_sph)

    # --- Redshift filter ---
    #this one will finally filter over the redshifts with error so the ones now outside
    #the redshift bin will also be filtered out. This will not be a lot
    d_coords_sph= filters.redshift_filter(d_coords_sph)

    # --- Luminosity filter ---
    # Applies apparent magnitude cut
    d_coords_sph = filters.luminosity_filter(d_coords_sph)

    


    data_size = np.shape(d_coords_sph)[0]
    if data_size < 2:
        print("Empty slice after redshift filtering.")
        return
    
    print("Data size after filtering:", data_size)
    
    # --- Correlation ---
    
    correlation = gal_cor.correlation_tools_pycorr(
            coordinates=d_coords_sph,
            cosmology=cosmo_real,
            cfg=cfg
    
    )
    
    ls=correlation.ls
    survey_density = correlation.galaxy_density
    print(dir(ls))
    print(np.shape(ls))
    print("Survey density :", survey_density)
    
    
    # --- Save output using HDF5 ---

    # Construct nested output directory structure
    output_dir = os.path.join(cfg.paths.output_directory, simulation)

    # Make sure it exists (this creates all intermediate subdirectories)
    os.makedirs(output_dir, exist_ok=True)

    # Build the full file path
    output_filename = os.path.join(
        output_dir,
        f"single_slice_snapshot_{redshift_number}.hdf5"
    )

    print("Saving results to:", output_filename)


    with h5py.File(output_filename, "w") as f:

        # --- Unitless arrays / scalars ---
        f.create_dataset("distance_type", data=cfg.distance.type)
        f.create_dataset("bins", data=correlation.bins)
        f.create_dataset("oversampling_factor", data=cfg.random_catalog.oversampling)
        f.create_dataset("random_size", data=correlation.n_random)
        f.create_dataset("ls_avg", data=ls_avg)
        f.create_dataset("ls_std", data=ls_std)

        # --- Arrays / scalars with units ---
        f.create_dataset("slice_thickness", data=cosmo.delta_dr.value)
        f["slice_thickness"].attrs["units"] = str(cosmo.delta_dr.units)

        f.create_dataset("survey_density", data=survey_density.value)
        f["survey_density"].attrs["units"] = str(survey_density.units)

        f.create_dataset("bao", data=correlation.bao.value)
        f["bao"].attrs["units"] = str(correlation.bao.units)

        f.create_dataset("min", data=correlation.min.value)
        f["min"].attrs["units"] = str(correlation.min.units)

        f.create_dataset("max", data=correlation.max.value)
        f["max"].attrs["units"] = str(correlation.max.units)

        f.create_dataset("bin_centers", data=correlation.bin_centers.value)
        f["bin_centers"].attrs["units"] = str(correlation.bin_centers.units)

        f.create_dataset("bin_width", data=correlation.bin_width.value)
        f["bin_width"].attrs["units"] = str(correlation.bin_width.units)

    print("Saved single-slice Landy-Szalay histogram to", output_filename)
    return output_filename #for plotting in the same pipeline!

