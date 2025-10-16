from swiftsimio import load
import numpy as np
import psutil
import os
import time
import gc
import threading

# Our own modules
import baoflamingo.galaxy_correlation as gal_cor
import baoflamingo.filtering as flt
import baoflamingo.statistics as stat
import baoflamingo.cosmology as cs
from baoflamingo.coordinates import coordinate_tools

# --- Memory monitor ---
def monitor_memory(interval=30):
    """Print memory usage every `interval` seconds."""
    process = psutil.Process(os.getpid())
    while True:
        mem_gb = process.memory_info().rss / (1024 ** 3)
        print(f"[MEMORY MONITOR] {mem_gb:.3f} GB")
        time.sleep(interval)

def run_pipeline_single(cfg):
    """Main pipeline using the YAML config object."""
    
    # Start memory monitor
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
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
        n_sigma=2
    )
    print("Cosmology set up")
    del metadata
    gc.collect()
    
    print("Slice thickness:", cosmo.delta_dr, "Mpc")
    
    # --- Observer position ---
    if cosmo.plus_dr < 0.5 * box_size_float:
        complete_sphere = True
        shift = np.random.uniform(-0.5*box_size_float, 0.5*box_size_float, size=3)
        observer = centre.copy()
        max_angle_plus_dr = 0
        print("Complete spherical slice is possible")
    else:
        complete_sphere = False
        min_x = np.ceil(cosmo.plus_dr)
        max_angle_plus_dr = np.arcsin(box_size_float / (2 * cosmo.plus_dr))
        max_x = box_size_float + cosmo.minus_dr * np.cos(max_angle_plus_dr)
        shift_observer_xyz = np.random.uniform(low=min_x, high=max_x, size=1)
        x_y_z_list = np.random.randint(3, size=1)
        observer = centre.copy()
        observer[x_y_z_list] = shift_observer_xyz
        shift = np.array([0, 0, 0])
        print("No complete spherical slice is possible")
    
    # --- Filtering ---
    filters = flt.filtering_tools(
        soap_file=data,
        cosmology=cosmo,
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
    radial_luminosity_mask = filters.radial_luminosity(d_coords_sph)
    gc.collect()
    #overwrite to save memory
    d_coords_sph = d_coords_sph[radial_luminosity_mask][:, 1:]
    
    data_size = np.shape(d_coords_sph)[0]
    if data_size < 2:
        print("Empty slice after redshift filtering.")
        return
    
    print("Data size after filtering:", data_size)
    
    # --- Correlation ---
    n_random = int(cfg.random_catalog.oversampling * data_size)
    correlation = gal_cor.correlation_tools_treecorr(
    cosmology=cosmo,
    min_distance=min_distance, max_distance=max_distance, n_random=n_random,
    max_angle=max_angle_plus_dr, complete_sphere=complete_sphere,
    bins=bins, distance_type=distance_type, seed=seed_random,
    variance_method=cfg.statistics.variance_method,n_patches=cfg.statistics.n_patches)
    
    mean_ls,std_ls = correlation.landy_szalay(coords=d_coords_sph)
    survey_density = correlation.galaxy_density(data_size)
    print("Survey density (#/deg^2):", survey_density)
    
    
    # --- Save output ---
    output_filename = f"single_slice_{cfg.distance.type}_{safe_simulation}_snapshot_{number}.npz"
    np.savez(
        output_filename,
        bin_centers=correlation.bin_centers,
        bin_width=correlation.bin_width,
        bins=correlation.bins,
        thickness_slice=cosmo.delta_dr,
        survey_density=survey_density,
        oversampling_factor=cfg.random_catalog.oversampling,
        random_size=correlation.n_random,
        bao_angle=correlation.bao_angle,
        bao_distance=cosmo.bao_distance,
        min_distance=cfg.distance.min,
        max_distance=cfg.distance.max,
        ls_avg=mean_ls,
        ls_std=std_ls
    )
    
    print("Saved single-slice Landy-Szalay histogram to", output_filename)



