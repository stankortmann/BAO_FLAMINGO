from swiftsimio import load
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as ss
import unyt as u
import psutil
import os
import time
import gc
import threading
import yaml

# Now our own modules
import baoflamingo.galaxy_correlation as gal_cor
import baoflamingo.filtering as flt
import baoflamingo.statistics as stat
import baoflamingo.cosmology as cs



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


# --- Paths ---
directory = "/cosma8/data/dp004/flamingo/Runs/"
simulation = "L2800N5040/HYDRO_FIDUCIAL"
safe_simulation = simulation.replace("/", "_")
soap_hbt_path = directory + simulation + "/SOAP-HBT/halo_properties_00"
redshift_path = directory + simulation + "/output_list.txt"

# --- Snapshot choice ---
redshift_list = np.loadtxt(redshift_path, skiprows=1)
number = 72  # which snapshot to load
b = ""       # to match filename structure if needed

# --- Distance range ---
min_distance = 50   # Mpc
max_distance = 200 # Mpc
distance_type= 'angular' #'angular'  or 'euclidean' 

# --- Random catalog ---
seed_random = 12345
oversampling = 2

# --- Filtering ---
central_filter = False
stellar_mass_filter = False
stellar_mass_cutoff = 90
luminosity_filter_switch = True
mr = 19
band = 'r'

# --- Plotting/statistics ---
bins = 200
leafsize = 100

# --- Load snapshot ---
data = load(soap_hbt_path + b + str(int(number)) + ".hdf5")
print("this is file", str(int(number)), "loaded")

# --- Cosmology ---
metadata = data.metadata
redshift = redshift_list[number]
print("this snapshot is at redshift z=", str(redshift))
cosmology = metadata.cosmology
box_size = metadata.boxsize.value
box_size_float = float(box_size[0])
middle = box_size_float / 2
centre = np.array([middle, middle, middle])

# --- Cosmology tools ---
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
print("The cosmology is set up, now metadata file can be closed")
del metadata
gc.collect()

print("The slice thickness is", cosmo.delta_dr, "Mpc")

# --- Observer position ---
if cosmo.plus_dr < 0.5 * box_size_float:
    complete_sphere = True
    shift_coordinates = np.random.uniform(
        low=-0.5*box_size_float,
        high=0.5*box_size_float,
        size=3
    )
    observer = centre.copy()
    shift = shift_coordinates

    #max_angle has to be defined for the correlation 
    max_angle_plus_dr=0
    print("In this snapshot a complete spherical slice is possible!!")
else:
    complete_sphere = False
    #minimimum x distance from box front
    min_x = np.ceil(cosmo.plus_dr)
    #maximum angle for imcomplete sphere
    max_angle_plus_dr = np.arcsin(box_size_float / (2 * cosmo.plus_dr)) #radians
    max_x = box_size_float + cosmo.minus_dr * np.cos(max_angle_plus_dr)
    
    shift_observer_xyz = np.random.uniform(low=min_x, high=max_x, size=1)
    x_y_z_list = np.random.randint(3, size=1)
    observer = centre.copy()
    observer[x_y_z_list] = shift_observer_xyz
    #no coordinate shift
    shift = np.array([0, 0, 0])
    print("In this snapshot NO complete spherical slice is possible")

# --- Filtering ---
filters = flt.filtering_tools(
    soap_file=data,
    cosmology=cosmo,
    central_filter=central_filter,
    stellar_mass_filter=stellar_mass_filter, stellar_mass_cutoff=stellar_mass_cutoff,
    luminosity_filter=luminosity_filter_switch, filter_band=band, m_cutoff=mr
)
gc.collect()
print("The filters are set up")

# --- Coordinate transformation ---
# d_coords must come from your filtering (e.g. galaxy positions)
d_coords = data.input_halos.halo_centre.value  # Example placeholder

coordinates = gal_cor.coordinate_tools(
    coordinates=d_coords,
    box_size=box_size_float,
    complete_sphere=complete_sphere,
    observer=observer,
    shift=shift
)


d_coords_sph = coordinates.spherical
del d_coords # deleting for memory purposes!


# --- Apply radial & luminosity mask ---
radial_luminosity_mask = filters.radial_luminosity(d_coords_sph)
gc.collect()
d_coords_sph = d_coords_sph[radial_luminosity_mask][:, 1:]

data_size = np.shape(d_coords_sph)[0]
if data_size < 2:
    print("Empty slice after redshift filtering.")
    exit()

print("The data size is", data_size)

# --- Correlation ---
n_random = int(oversampling * data_size)


############ NEW CORRELATION TOOL #################
correlation = gal_cor.correlation_tools_treecorr(
    cosmology=cosmo,
    min_distance=min_distance, max_distance=max_distance, n_random=n_random,
    max_angle=max_angle_plus_dr, complete_sphere=complete_sphere,
    bins=bins, distance_type=distance_type, seed=seed_random,
    variance_method='jackknife',n_patches=100)

mean_ls,std_ls = correlation.landy_szalay(coords=d_coords_sph)
survey_density = correlation.galaxy_density(data_size)
print("survey density is ",survey_density.to(1/u.arcmin**2))


# --- Save output ---
output_filename = f"single_slice_{distance_type}_{safe_simulation}_snapshot_{number}.npz"
np.savez(
    output_filename,
    bin_centers=correlation.bin_centers,
    bin_width=correlation.bin_width,
    bins=correlation.bins,
    thickness_slice=cosmo.delta_dr,
    survey_density=survey_density,
    oversampling_factor=oversampling,
    random_size=correlation.n_random,
    bao_angle=correlation.bao_angle,
    bao_distance=cosmo.bao_distance,
    min_distance=min_distance,
    max_distance=max_distance,
    ls_avg=mean_ls,
    ls_std=std_ls
)

print("Saved single-slice Landy-Szalay histogram.")
print("Data saved in", output_filename)
exit()
