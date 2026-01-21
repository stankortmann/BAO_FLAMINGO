import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology as cosmo_fiducial
from mcfit import P2xi

from baoflamingo.template import template_CAMB  
from baoflamingo.cosmology import cosmo_tools
import unyt as u

ic_file= "/cosma8/data/dp004/jlvc76/FLAMINGO/ICs/L1000N1800/HYDRO_FIDUCIAL/input_powerspec.txt"
data_ic=np.loadtxt(ic_file)
k_sim_lin=data_ic[:,0]  #Mpc^-1
pk_sim_lin=data_ic[:,1]  #Mpc^3

# --- Define s array ---
s_array = np.linspace(125, 200, 50)  # Mpc (comoving)

#now use the power spectrum of the simulation to check if it matches
# --- Simulated P(k) ---
data=np.loadtxt("/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/power_spectra/power_matter_0104.txt")
#z_eff=data[0,0]
"""
k_sim_lin  = data[:, 1]   # Mpc^-1
pk_sim_lin= data[:, 2]   # Mpc^3
"""
# Build log-spaced k grid
k_sim_log = np.logspace(np.log10(k_sim_lin.min()),
                    np.log10(k_sim_lin.max()),
                    2048)

# Interpolate P(k) onto log grid
pk_sim_log = np.interp(k_sim_log, k_sim_lin, pk_sim_lin)

# --- Effective redshift ---
z_eff = 100  # example

# --- Initialize the BAO template ---


cosmo_fiducial.setCosmology("illustris")
cosmo=cosmo_tools(box_size=1000*u.Unit("Mpc"),
    constants=cosmo_fiducial.current_cosmo,
    redshift=z_eff,
    redshift_bin_width=0.2)
print("cosmo is set up!")
template = template_CAMB(cosmo=cosmo, effective_redshift=z_eff, s_array=s_array, ell_list=[0,2], non_linear=True)

# --- Get multipoles ---
xi_dict = template.get_multipoles()
xi0 = xi_dict[0]  # monopole
xi2 = xi_dict[2]  # quadrupole (should be ~0 for ΛCDM)
print(f"xi0 shape:{np.shape(xi0)}")
print(f"xi2 shape:{np.shape(xi2)}")
s=template.s


P2xi_obj = P2xi(k=k_sim_log,l=0, lowring=False)
r,xi = P2xi_obj(pk_sim_log, extrap=True)
#multiply by h because s is expected in Mpc/h
xi_sim = np.interp(s*0.681, r, xi)
# --- Plot the results ---
plt.figure(figsize=(8,5))
#plt.plot(s, xi0, label=r'Monopole $ξ_0(s)$ CAMB template', marker='o',color='blue')
#plt.plot(s_array, xi2, label='Quadrupole ξ2(s)', marker='x')



plt.plot(s, xi_sim, label=r'Monopole $ξ_0(s)$ Power spectrum simulation', color='red')
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
print(f"bao peak template:{s[np.argmax(xi0)]}")
print(f"bao peak simulation:{s[np.argmax(xi_sim)]}")




exit()