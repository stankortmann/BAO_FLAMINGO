from swiftsimio import load
import numpy as np
import psutil
import os
import time
import gc
import threading

# Own modules
import galaxy_correlation as gal_cor
import filtering as flt
import statistics as stat
import cosmology as cs


# ===========================
# MEMORY MONITOR
# ===========================
def monitor_memory(interval=30):
    """Print memory usage every `interval` seconds."""
    process = psutil.Process(os.getpid())
    while True:
        mem_gb = process.memory_info().rss / (1024 ** 3)
        print(f"[MEMORY MONITOR] {mem_gb:.3f} GB")
        time.sleep(interval)


# Start memory monitor in background
monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
monitor_thread.start()


# ===========================
# CONFIGURATION
# ===========================
directory = "/cosma8/data/dp004/flamingo/Runs/"
simulation = "L2800N5040/HYDRO_FIDUCIAL"
safe_simulation = simulation.replace("/", "_")

soap_hbt_path = f"{directory}{simulation}/SOAP-HBT/halo_properties_00"
redshift_path = f"{directory}{simulation}/output_list.txt"

redshift_list = np.loadtxt(redshift_path, skiprows=1)
snapshot_numbers = [72]   # list of snapshots to process

n_slices = 20             # number of observer positions per snapshot
min_distance = 0          # Mpc
max_distance = 220        # Mpc
distance_type = "angular" # 'angular' or 'euclidean'
seed_random = 12345
oversampling = 2

central_filtering = False
mass_percentile = 90
mr = 19                   # luminosity cutoff (apparent mag)
bins = 200
leafsize = 100

n_bootstrap = int(1e6)
low_per = 16
high_per = 84


# ===========================
# MAIN LOOP OVER SNAPSHOTS
# ===========================
for a in snapshot_numbers:
    # File prefix
    b = "0" if a < 10 else ""

    # --- Load snapshot ---
    data = load(f"{soap_hbt_path}{b}{int(a)}.hdf5")
    print(f"Snapshot {a} loaded.")

    # --- Cosmology ---
    metadata = data.metadata
    redshift = redshift_list[a]
    cosmology = metadata.cosmology
    box_size = cosmology.boxsize.value if hasattr(cosmology, "boxsize") else metadata.boxsize.value
    box_size_float = float(box_size[0])
    middle = box_size_float / 2
    centre = np.array([middle, middle, middle])

    cosmo = cs.cosmo_tools(
        H0=cosmology.H0.value,
        Omega_m=cosmology.Om0,
        Omega_b=cosmology.Ob0,
        Omega_lambda=cosmology.Ode0,
        Tcmb=cosmology.Tcmb0.value,
        Neff=cosmology.Neff,
        redshift=redshift,
        n_sigma=2
    )
    print(f"Redshift z = {redshift:.3f}")
    print(f"Comoving distance: {cosmo.comoving_distance:.3f} Mpc")
    print(f"Slice thickness: {cosmo.delta_dr:.3f} Mpc")
    del metadata
    gc.collect()

    # --- Observer placement (complete vs incomplete sphere) ---
    if cosmo.plus_dr < 0.5 * box_size_float:
        complete_sphere = True
        shift_coordinates = np.random.uniform(
            low=-0.5 * box_size_float,
            high=0.5 * box_size_float,
            size=(n_slices, 3)
        )
        print("Complete spherical slice possible.")
    else:
        complete_sphere = False
        min_x = np.ceil(cosmo.plus_dr)
        max_angle_plus_dr = np.arcsin(box_size_float / (2 * cosmo.plus_dr))
        max_x = box_size_float + cosmo.minus_dr * np.cos(max_angle_plus_dr)
        shift_observer_xyz = np.random.uniform(low=min_x, high=max_x, size=n_slices)
        x_y_z_list = np.random.randint(3, size=n_slices)
        print("No complete spherical slice possible.")

    # --- Filtering ---
    filters = flt.filtering_tools(
        soap_file=data,
        cosmology=cosmo,
        central_filter=False,
        stellar_mass_filter=False, stellar_mass_cutoff=0,
        luminosity_filter=True, filter_band='r', m_cutoff=mr
    )
    gc.collect()

    d_coords = data.input_halos.halo_centre.value
    luminosity_mask = filters.luminosity_filter()
    d_coords = d_coords[luminosity_mask]
    print(f"Number of galaxies after filtering: {d_coords.shape[0]}")

    # ===========================
    # LOOP OVER SLICES
    # ===========================
    all_ls = []
    all_data_size = []
    correlation = None

    for i in range(n_slices):
        # --- Observer position ---
        if complete_sphere:
            observer = centre.copy()
            shift = shift_coordinates[i]
        else:
            observer = centre.copy()
            observer[x_y_z_list[i]] = shift_observer_xyz[i]
            shift = np.array([0, 0, 0])

        # --- Coordinate transform ---
        coordinates = gal_cor.coordinate_tools(
            coordinates=d_coords,
            box_size=box_size_float,
            complete_sphere=complete_sphere,
            observer=observer,
            shift=shift
        )
        d_coords_sph = coordinates.spherical

        # --- Radial filtering ---
        spherical_mask = filters.radial_filter(d_coords_sph)
        d_coords_sph = d_coords_sph[spherical_mask][:, 1:]  # keep theta, phi only

        data_size = d_coords_sph.shape[0]
        if data_size < 2:
            print(f"Empty slice {i}, skipping.")
            continue

        print(f"Slice {i+1}/{n_slices}: data size = {data_size}")

        # --- Initialize correlation class (once) ---
        if correlation is None:
            n_random = int(oversampling * data_size)
            correlation = gal_cor.correlation_tools(
                box_size=1000.0, radius=427.0, max_angle_plus_dr=10.0,
                min_distance=min_distance, max_distance=max_distance,
                bao_distance=cosmo.bao_distance,
                complete_sphere=complete_sphere,
                leafsize=leafsize,
                seed=seed_random, n_random=n_random,
                bins=bins, distance_type=distance_type
            )

        # --- Landy–Szalay ---
        hist_ls = correlation.landy_szalay(data_coords=d_coords_sph)
        all_ls.append(hist_ls)
        all_data_size.append(data_size)
        print(f"Slice {i+1} done.")

    # ===========================
    # STATISTICS ACROSS SLICES
    # ===========================
    all_ls = np.array(all_ls)
    all_data_size = np.array(all_data_size)
    stats_obj = stat.stat_tools(
        all_ls=all_ls,
        weights=all_data_size,
        random_seed=12345
    )

    mean_ls, std_ls = stats_obj.weighted_avg_and_std()

    mean_bs, std_bs, ci_low, ci_high = stats_obj.bootstrap_slices(
        n_bootstrap=n_bootstrap,
        percentiles=[low_per, high_per]
    )

    # ===========================
    # SAVE RESULTS
    # ===========================
    output_filename = f"multiple_slices_{distance_type}_{safe_simulation}_snapshot_{b}{int(a)}.npz"
    np.savez(
        output_filename,
        bin_centers=correlation.bin_centers,
        bin_width=correlation.bin_width,
        bins=correlation.bins,
        slices=n_slices,
        thickness_slice=cosmo.delta_dr,
        data_sizes=all_data_size,
        oversampling=oversampling,
        random_size=correlation.n_randoms,
        min_angle=correlation.min_angle,
        max_angle=correlation.max_angle,
        bao_angle=correlation.bao_angle,
        bao_distance=cosmo.bao_distance,
        max_distance=max_distance,
        ls_avg=mean_ls,
        ls_std=std_ls,
        ls_avg_bs=mean_bs,
        ls_std_bs=std_bs,
        n_bootstrap=n_bootstrap,
        ci_low_bs=ci_low,
        ci_high_bs=ci_high,
        low_per=low_per,
        high_per=high_per
    )

    print(f"Saved averaged Landy–Szalay histogram for snapshot {a} to {output_filename}")

exit()
