import numpy as np
from collections import Counter
import math

# ============================================================
# USER PARAMETERS (EDIT THESE)
# ============================================================

# Box side length
L = 1000.0  # change if needed

# Spherical shell parameters

r_in = 2100                  # inner radius (edit)
r_out = 2300                  # outer radius (edit)
Cx, Cy, Cz = L-1-r_out,0.5*L,0.5*L # center of shell (can be outside the box)
# How many periodic boxes to add in each direction
# Example: n_left_x = 1 means we include box at x = [-L, 0)
#          n_right_x = 2 means boxes at x = [L,2L)
n_left_x  = 0
n_right_x = 0
n_left_y  = 1
n_right_y = 0
n_left_z  = 0
n_right_z = 0

# Monte Carlo sampling resolution
N_samples = 100000  # increase later for precision

# ============================================================
# END USER PARAMETERS
# ============================================================

# Build the list of periodic image shifts
shift_list = []
for nx in range(-n_left_x, n_right_x + 1):
    for ny in range(-n_left_y, n_right_y + 1):
        for nz in range(-n_left_z, n_right_z + 1):
            shift_list.append((nx, ny, nz))
shift_list = np.array(shift_list, dtype=int)
n_images = len(shift_list)

print(f"Using {n_images} periodic boxes (including original).")

# Compute image centers for the spherical shell
base_center = np.array([Cx, Cy, Cz], dtype=float)
image_centers = base_center + shift_list * L  # shape (n_images, 3)

# Generate Monte Carlo sample points inside the *original* box
points = np.random.uniform(0.0, L, size=(N_samples, 3))

# Count multiplicity: how many images contain each sample point
multiplicity = np.zeros(N_samples, dtype=int)

for center in image_centers:
    disp = points - center.reshape(1, 3)
    dist2 = np.sum(disp * disp, axis=1)
    inside = (dist2 >= r_in * r_in) & (dist2 <= r_out * r_out)
    multiplicity += inside.astype(int)

# Build histogram
counts = Counter(multiplicity.tolist())
max_mult = max(counts.keys()) if len(counts) > 0 else 0

fractions = np.array([counts.get(k, 0) / N_samples for k in range(max_mult + 1)])
f_covered = 1.0 - fractions[0] if len(fractions) > 0 else 0.0
f_overlap2 = fractions[2:].sum() if len(fractions) > 2 else 0.0

# Print results
print("\n=========== RESULTS ===========")
print(f"Fraction covered at least once: {f_covered*100:.6f}%")
print(f"Fraction covered >= 2 times   : {f_overlap2*100:.6f}%")
print("\nMultiplicity histogram:")
for k, frac in enumerate(fractions):
    print(f"  {k}: {frac*100:.6f}%")

# Return results as dictionary (useful if importing)
results = {
    "L": L,
    "center": (Cx, Cy, Cz),
    "r_in": r_in,
    "r_out": r_out,
    "N_samples": N_samples,
    "n_images": n_images,
    "fractions": fractions,
    "fraction_covered": f_covered,
    "fraction_overlap": f_overlap2,
}
