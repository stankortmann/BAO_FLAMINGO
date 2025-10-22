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
import baoflamingo.galaxy_correlation as gal_cor
import baoflamingo.filtering as flt
import baoflamingo.statistics as stat
import baoflamingo.cosmology as cs
from baoflamingo.coordinates import coordinate_tools

# --- Memory and monitor ---

def monitor_system(interval=30):
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
    safe_simulation = simulation.replace("/", "_")
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
    box_size_float = float(metadata.boxsize.value[0])
    centre = np.array([box_size_float / 2] * 3)
    
    cosmo = cs.cosmo_tools(
        box_size=box_size_float,
        H0=cosmology.H0.value,
        Omega_m=cosmology.Om0,
        Omega_b=cosmology.Ob0,
        Omega_lambda=cosmology.Ode0,
        Tcmb=cosmology.Tcmb0.value,
        Neff=cosmology.Neff,
        redshift=redshift,
        n_sigma=cfg.slicing.n_sigma
    )
    print("Cosmology set up")
    print("Radius is",cosmo.comoving_distance)
    del metadata
    gc.collect()
    
    print("Slice thickness:", cosmo.delta_dr)
    
    # --- Observer position ---
    if cosmo.plus_dr < 0.5 * box_size_float:
        complete_sphere = True
        shift = np.random.uniform(-0.5*box_size_float, 0.5*box_size_float, size=3)
        max_angle_plus_dr = 0
        observer = centre.copy()
        
        print("Complete spherical slice is possible")
    
    #We have to take a look at this!
    
    else:
        complete_sphere = False
        shift = np.random.uniform(-0.5*box_size_float, 0.5*box_size_float, size=3)
        #we have to do extra slicing with this one!
        max_angle_plus_dr = np.arcsin(box_size_float / (2 * cosmo.plus_dr))*u.rad
        observer = centre.copy()
        observer[0] += cosmo.comoving_distance.value
        print("No complete spherical slice is possible")
    
    # --- Filtering ---
    filters = flt.filtering_tools(
        soap_file=data,
        cosmology=cosmo,
        complete_sphere=complete_sphere,
        max_angle_incomplete=max_angle_plus_dr,
        central_filter=cfg.filters.central_filter,
        stellar_mass_filter=cfg.filters.stellar_mass_filter,
        stellar_mass_cutoff=cfg.filters.stellar_mass_cutoff,
        luminosity_filter=cfg.filters.luminosity_filter,
        filter_band=cfg.filters.band,
        m_cutoff=cfg.filters.m_cutoff
    )
    gc.collect()
    print("Filters set up")
    
    # --- Coordinate transformation ---
    d_coords = data.input_halos.halo_centre.value
    coordinates = coordinate_tools(
        coordinates=d_coords,
        box_size=box_size_float,
        complete_sphere=complete_sphere,
        observer=observer,
        shift=shift
    )
    d_coords_sph = coordinates.spherical

    #memory efficiency to delete original d_coords from memory
    del d_coords
    gc.collect()
    
    # --- Apply radial & luminosity mask ---
    #radial_luminosity_mask = filters.radial_luminosity(d_coords_sph)
    radial_luminosity_mask=filters.radial_luminosity(d_coords_sph)
    gc.collect()
    #overwrite to save memory, change this
    d_coords_sph = d_coords_sph[radial_luminosity_mask][:, 1:]
    
    data_size = np.shape(d_coords_sph)[0]
    if data_size < 2:
        print("Empty slice after redshift filtering.")
        return
    
    print("Data size after filtering:", data_size)
    
    # --- Correlation ---
    n_random = int(cfg.random_catalog.oversampling * data_size)
    correlation = gal_cor.correlation_tools_treecorr_test(
            cosmology=cosmo,
            #include unyt units
            min_distance=cfg.distance.min*u.Mpc,
            max_distance=cfg.distance.max*u.Mpc,

            n_random=n_random,
            max_angle=max_angle_plus_dr,
            complete_sphere=complete_sphere,
            bins=cfg.plotting.bins, 
            distance_type=cfg.distance.type, 
            seed=cfg.random_catalog.seed,
            variance_method=cfg.statistics.variance_method,
            n_patches=cfg.statistics.n_patches
    
    )
    
    avg_ls,std_ls = correlation.landy_szalay(coords=d_coords_sph)
    survey_density = correlation.galaxy_density(data_size)
    print("Survey density :", survey_density)
    
    
    # --- Save output using HDF5 ---
    output_filename = os.path.join(
        cfg.paths.output_directory,
        f"single_slice_{safe_simulation}_snapshot_{redshift_number}.hdf5"
    )

    with h5py.File(output_filename, "w") as f:

        # --- Unitless arrays / scalars ---
        f.create_dataset("distance_type", data=cfg.distance.type)
        f.create_dataset("bins", data=correlation.bins)
        f.create_dataset("oversampling_factor", data=cfg.random_catalog.oversampling)
        f.create_dataset("random_size", data=correlation.n_random)
        f.create_dataset("avg_ls", data=avg_ls)
        f.create_dataset("std_ls", data=std_ls)

        # --- Arrays / scalars with units ---
        f.create_dataset("thickness_slice", data=cosmo.delta_dr.value)
        f["thickness_slice"].attrs["units"] = str(cosmo.delta_dr.units)

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

