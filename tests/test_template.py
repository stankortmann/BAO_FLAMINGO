import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology as cosmo_fiducial

from baoflamingo.template import template_CAMB  
from baoflamingo.cosmology import cosmo_tools
import unyt as u


# --- Define s array ---
s_array = np.linspace(130, 200, 30)  # Mpc (comoving)

# --- Effective redshift ---
z_eff = 0.5  # example

# --- Initialize the BAO template ---


cosmo_fiducial.setCosmology("illustris")
cosmo=cosmo_tools(box_size=1000*u.Unit("Mpc"),
    constants=cosmo_fiducial.current_cosmo,
    redshift=z_eff,
    redshift_bin_width=0.2)
print("cosmo is set up!")
template = template_CAMB(cosmo=cosmo, effective_redshift=z_eff, s_array=s_array, ell_list=[0,2])

# --- Get multipoles ---
xi_dict = template.get_multipoles()
xi0 = xi_dict[0]  # monopole
xi2 = xi_dict[2]  # quadrupole (should be ~0 for ΛCDM)
print(f"xi0 shape:{np.shape(xi0)}")
print(f"xi2 shape:{np.shape(xi2)}")
s=template.s
plt.figure(figsize=(8,5))
plt.plot(s, xi0, label='Monopole ξ0(s)', marker='o')
#plt.plot(s_array, xi2, label='Quadrupole ξ2(s)', marker='x')
plt.xlabel(r'$s \, [\mathrm{Mpc}]$')
plt.ylabel(r'$\xi_\ell(s)$')
plt.title(f'BAO template at z={z_eff}')
plt.legend()
plt.grid(True)

# --- Save figure ---
plt.savefig("BAO_template.png", dpi=300)   # saves as PNG
# plt.savefig("BAO_template.pdf")          # or save as PDF

plt.show()
print("done")
print(f"bao peak:{s[np.argmax(xi0)]}")
exit()