from baoflamingo.coordinates import coordinate_tools
import numpy as np
import matplotlib.pyplot as plt

N_data = 2000
rng=np.random.default_rng(42)
# --- Synthetic galaxy redshift distribution ---

test=2


if test==0:
    z_data = np.concatenate([
        rng.normal(0.5, 0.1, int(0.8 * N_data)),
        rng.exponential(0.3, int(0.2 * N_data))
    ])

    z_data = z_data[(z_data > 0.0) & (z_data < 2.0)]

if test==1:
    # Three populations with declining weights
    z_low = np.random.normal(0.3, 0.05, size=300)   # many galaxies
    z_mid = np.random.normal(0.5, 0.05, size=150)   # fewer galaxies
    z_high = np.random.normal(0.7, 0.07, size=50)   # rare galaxies

    # Concatenate into a single array
    z_data = np.concatenate([z_low, z_mid, z_high])

    # Keep within physical range 0 < z < 1
    z_data = z_data[(z_data > 0) & (z_data < 1)]

if test == 2:
    z_data = np.loadtxt('redshift_data.txt')

# --- Generate random redshifts from n(z) ---
n_random = 5 * len(z_data)
z_rand, z_grid, n_z = coordinate_tools.random_redshifts_from_data_cdf(
    data_z=z_data, 
    n_random=n_random, 
    smoothing=0.01, 
    rng=rng
    )


# --- Plot results ---
plt.figure(figsize=(7, 5))
plt.hist(z_data, bins=40, density=True, histtype='step', lw=2, label='Data n(z)')
plt.plot(z_grid, n_z, 'r-', lw=2, label='Smoothed PDF')
plt.hist(z_rand, bins=40, density=True, histtype='step', lw=2, label='Random n(z)')
plt.xlabel('Redshift z')
plt.ylabel('Normalized n(z)')
plt.legend()
plt.title("draw_random_redshifts_from_data() test")

# Save plot
plt.tight_layout()
plt.savefig("n_z_test.png", dpi=200)
plt.close()

# --- Print quick stats ---
print("Saved plot: n_z_test.png")
print(f"Mean z_data = {np.mean(z_data):.3f}, Mean z_rand = {np.mean(z_rand):.3f}")
print(f"Std  z_data = {np.std(z_data):.3f}, Std  z_rand = {np.std(z_rand):.3f}")