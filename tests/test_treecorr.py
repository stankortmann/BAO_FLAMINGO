import numpy as np
import unyt as u
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from baoflamingo.cosmology import cosmo_tools
from baoflamingo.coordinates import coordinate_tools
from baoflamingo.galaxy_correlation import correlation_tools_treecorr_test

# --- Step 1: define a simple cosmology ---
cosmo = cosmo_tools(
    box_size=1000,    # Mpc
    H0=70,
    Omega_m=0.3,
    Omega_b=0.048,
    Omega_lambda=0.7,
    Tcmb=2.725,
    Neff=3.046,
    redshift=0.5,
    n_sigma=3
)

# --- Step 2: make synthetic galaxy coordinates ---
# small clustered sphere for test
seed=101
rng = np.random.default_rng(seed)
n_gal = int(1e4) #amount of galaxies
theta = np.arccos(1 - 2 * rng.random(n_gal))
phi = 2 * np.pi * rng.random(n_gal)
coords_original=np.column_stack([theta,phi])


# artificially cluster a subset
r_bump=50 #mpc
D_c=(cosmo.comoving_distance)
delta_theta = 2 * np.arcsin(r_bump / (2 * D_c))
N_pairs = int(0.025*n_gal)  # number of injected bump pairs

# pick random "seed" positions
theta0 = np.arccos(1 - 2 * rng.random(N_pairs))
phi0 = 2 * np.pi * rng.random(N_pairs)

# random azimuthal directions around each seed
alpha = 2 * np.pi * rng.random(N_pairs)

# compute companion positions at angular distance delta_theta
theta1 = np.arccos(np.cos(theta0)*np.cos(delta_theta) +
                   np.sin(theta0)*np.sin(delta_theta)*np.cos(alpha))
phi1 = phi0 + np.arctan2(np.sin(alpha)*np.sin(delta_theta)*np.sin(theta0),
                         np.cos(delta_theta) - np.cos(theta0)*np.cos(theta1))

# concatenate with the base sample
coords_bump = np.vstack([
    np.column_stack([theta0, phi0]),
    np.column_stack([theta1, phi1])
])
#combine sthem
coords_total = np.vstack([coords_original, coords_bump])


# --- Step 3: initialize correlation tool ---
corr = correlation_tools_treecorr_test(
    cosmology=cosmo,
    min_distance=20,     # Mpc
    max_distance=100,   # Mpc
    n_random=20*n_gal,
    bins=50,
    distance_type='euclidean',
    seed=seed,
    variance_method='jackknife',
    n_patches=60
)

# --- Step 4: compute Landy–Szalay ---
mean_xi, std_xi = corr.landy_szalay(coords_total)

# --- Step 5: inspect results ---
print("Mean xi:", mean_xi)
print("Std xi:", std_xi)
print("Bin centers:", corr.bin_centers)

# --- Step 6: Fit Spline and try to find bump ---
spline = UnivariateSpline(corr.bin_centers, mean_xi, s=1)
baseline = spline(corr.bin_centers)

# compute residual
xi_residual = mean_xi - baseline



peak_idx = np.argmax(xi_residual)
r_bao = corr.bin_centers[peak_idx]
xi_bao = xi_residual[peak_idx]


print(r_bao)



plt.figure(figsize=(7,4))
plt.plot(corr.bin_centers, baseline, color='r', linestyle='--', label='Smooth baseline')
plt.errorbar(
    corr.bin_centers, mean_xi,
    yerr=std_xi*5,
    fmt='o',             # marker type: circle
    markersize=3,        # small marker
    elinewidth=1,        # thickness of error bars
    capsize=2,           # small horizontal cap on error bars
    color='blue',
    label='ξ(r)'
)

r_bump_ang=coordinate_tools.chord_to_angular_radians(r_bump/D_c)
plt.axvline(r_bump_ang, color="g", linestyle="--")
#plt.xscale('log')
plt.xlabel(f"r [{corr.bin_centers.units}]")
plt.ylabel("ξ(r)")
plt.title("Synthetic test correlation")
plt.grid(True)
plt.savefig("test_plot_radec.png", dpi=300)
plt.show()
exit()
