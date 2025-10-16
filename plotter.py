import numpy as np
import matplotlib.pyplot as plt

#own modules!
import statistics as stat
import smooth_fitting as sf

# ============================================================
# --- CONFIGURATION ---
# ============================================================

# Simulation parameters
simulation = "L2800N5040/HYDRO_FIDUCIAL"
redshift = 72

# Plot settings
angles=True
if angles:
    distance_type='angular' #or 'euclidean' 
else:
    distance_type='euclidean'
 
show_fit = False         # whether to fit & plot smooth models
save_plot = True         # whether to save the figure as PNG

# ============================================================
# --- FILENAME SETUP ---
# ============================================================

safe_simulation = simulation.replace("/", "_")
sim_name = f"{safe_simulation}_snapshot_{redshift}"


filename_histogram = f"single_slice_{distance_type}_{sim_name}.npz"

# PNG file name for output
png_filename = filename_histogram.replace(".npz", "_plot.png")

# ============================================================
# --- LOAD DATA ---
# ============================================================

try:
    data = np.load(filename_histogram)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {filename_histogram}")

bin_centers = data["bin_centers"]
bao_angle = data["bao_angle"]
ls_avg = data["ls_avg"]
print(ls_avg)
print(bin_centers)

# Optional: standard deviation if needed
# ls_std = data["ls_std_bs"]

# ============================================================
# --- SELECT DATA BASED ON ANGLE / SEPARATION RANGE ---
# ============================================================

# Define your range
angle_min, angle_max = 0, 100 # degrees (or Mpc if angular=False)

# Create a boolean mask for the range
mask = (bin_centers >= angle_min) & (bin_centers <= angle_max)

# Apply mask to your arrays
bin_centers_plot = bin_centers[mask]
ls_avg_plot = ls_avg[mask]
# ls_std_plot = ls_std[mask]  # if using std


# ============================================================
# --- PLOTTING ---
# ============================================================

plt.figure(figsize=(8, 6))

# Main correlation function
plt.plot( ls_avg_plot, label="Landy–Szalay", lw=1.8)

# Optional error bars (uncomment if you have std)
# plt.errorbar(bin_centers_plot, ls_avg_plot, yerr=ls_std_plot,
#              label="Landy–Szalay (error)", alpha=0.7, ecolor="r", fmt="o")

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

plt.xlabel("Angle (degrees)" if angles else "Separation (Mpc)")
plt.ylabel("Correlation")
plt.title(sim_name)
plt.legend()
plt.grid(alpha=0.3)

if save_plot:
    plt.tight_layout()
    plt.savefig(png_filename, dpi=300)
    print(f"Plot saved as '{png_filename}'")

plt.show()
