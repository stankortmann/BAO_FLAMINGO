import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import unyt as u
import h5py

#own modules!
import statistics as stat
import smooth_fitting as sf

# ============================================================
# --- CONFIGURATION ---
# ============================================================

# Simulation parameters
simulation = "L1000N3600/HYDRO_FIDUCIAL"
redshift = 69


 
show_fit = False         # whether to fit & plot smooth models
save_plot = True         # whether to save the figure as PNG

# ============================================================
# --- FILENAME SETUP ---
# ============================================================

safe_simulation = simulation.replace("/", "_")
sim_name = f"{safe_simulation}_snapshot_{redshift}"


filename_histogram = f"results_npz/single_slice_{sim_name}.hdf5"

# PNG file name for output
png_filename = filename_histogram.replace(".hdf5", "_plot.png")

# ============================================================
# --- LOAD DATA ---
# ============================================================


# Dictionary to hold the loaded arrays
data = {}

with h5py.File(filename_histogram, "r") as f:
    for key in f.keys():
        dset = f[key]
        if "units" in dset.attrs:  # if the dataset has units
            data[key] = dset[()] * u.Unit(dset.attrs["units"])
        else:  # unitless
            data[key] = dset[()]

bin_centers = data["bin_centers"]
bao_angle = data["bao"]
ls_avg = data["avg_ls"]
ls_std =data["std_ls"]
distance_type=data["distance_type"]


# Optional: standard deviation if needed
# ls_std = data["ls_std_bs"]

# ============================================================
# --- SELECT DATA BASED ON ANGLE / SEPARATION RANGE ---
# ============================================================

# Define your range
angle_min, angle_max = 0*u.Mpc, 250*u.Mpc # degrees (or Mpc if angular=False)
print(angle_min,bin_centers[0])
# Create a boolean mask for the range
mask = (bin_centers >= angle_min) & (bin_centers <= angle_max)

# Apply mask to your arrays
bin_centers_plot = bin_centers[mask]
ls_avg_plot = ls_avg[mask]
ls_std_plot=ls_std[mask]
# ls_std_plot = ls_std[mask]  # if using std


# ============================================================
# --- PLOTTING ---
# ============================================================

plt.figure(figsize=(8, 6))

# Main correlation function
plt.scatter(bin_centers_plot,ls_avg_plot, label="Landy–Szalay")
spline = UnivariateSpline(bin_centers_plot, ls_avg_plot, s=0.1)
baseline = spline(bin_centers_plot)

# compute residual
xi_residual = ls_avg_plot - baseline

mask_bao = (bin_centers.value >= bao_angle.value-10) &\
 (bin_centers.value <= bao_angle.value+10)

peak_idx = np.argmax(xi_residual)
r_bao = bin_centers_plot[peak_idx]
xi_bao = xi_residual[peak_idx]
plt.plot(bin_centers_plot, baseline, color='r', linestyle='--', label='Smooth baseline')

print(f"BAO peak at r ~ {r_bao:.2f} Mpc with ξ ~ {xi_bao:.4f}")
print('The expected BAO peak is at',bao_angle)

# Optional error bars (uncomment if you have std)
#plt.errorbar(bin_centers_plot, ls_avg_plot, yerr=ls_std_plot,
#           label="Landy–Szalay (error)", alpha=0.7, ecolor="r", fmt="x")

# Vertical line marking the BAO angle
plt.axvline(bao_angle, color="g", linestyle="--", label=f"BAO angle = {bao_angle:.2f}")

# ============================================================
# --- OPTIONAL FITTING ---
# ============================================================

if show_fit:
    w_power, w_poly = sf.fit_smooth_correlation(
        bin_centers, ls_avg, theta_min=None, theta_max=80
    )

    # Compute residuals for inspection
    diff_power = ls_avg - w_power
    diff_poly = ls_avg - w_poly

    # Plot fits and residuals
    plt.plot(bin_centers_plot, w_power[r_min:r_max], label="Power-law fit", color="black")
    plt.plot(bin_centers_plot, diff_power[r_min:r_max], label="Data - Smooth", color="orange")

# ============================================================
# --- LABELS & SAVE ---
# ============================================================

plt.xlabel(f"r [{bin_centers_plot.units}]")
plt.ylabel("ξ(r)")
plt.title(sim_name)
plt.legend()
plt.grid(alpha=0.3)

if save_plot:
    plt.tight_layout()
    plt.savefig(png_filename, dpi=300)
    print(f"Plot saved as '{png_filename}'")

plt.show()
