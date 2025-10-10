import numpy as np
import matplotlib.pyplot as plt
import formulae as fm

# --- Parameters ---
# Change this to the filename you want to plot
simulation="L2800N5040/HYDRO_FIDUCIAL"
safe_simulation = simulation.replace("/", "_") #for directory purposes
redshift=73
sim_name=safe_simulation+"_data_00"+str(redshift)

angular=False
if angular==True:
    ang="_angular_"
else:
    ang="_"
filename_histogram = "pdh"+ang+"avgs_"+sim_name+".npz"  # your saved .npz file

# --- Load the saved data ---
data = np.load(filename_histogram)

bin_centers = data['bin_centers']
bao_angle=data['bao_distance']
ls_avg=data['ls_avg'][0] #bootstrapping now used
#ls_std=data['ls_std_bs']

ls_std_plot=None



# --- Plotting ---
plt.figure(figsize=(8, 6))
begin=0
end=700
print(ls_avg)
ls_avg_plot=ls_avg[begin:end]
bin_centers_plot=bin_centers[begin:end]
#ls_std_plot=ls_std[begin:end]

"""
plt.errorbar(bin_centers_plot, ls_avg_plot,
yerr=ls_std_plot, label="Landy-Szalay data",
   alpha=0.7,ecolor='r')

"""
plt.plot(bin_centers_plot,ls_avg_plot)

"""


# --- Fitting the data 

w_power,w_poly = fm.fit_smooth_correlation(
    bin_centers,ls_avg,theta_min=None, theta_max=80)


#see if we want to return the fitted parameters later
#print("Fitted A =", A_fit, "gamma =", gamma_fit)

diff_power=ls_avg-w_power
diff_power_plot=diff_power[begin:end]
w_power_plot=w_power[begin:end]

diff_poly=ls_avg-w_poly
diff_poly_plot=diff_poly[begin:end]
w_poly_plot=w_poly[begin:end]

#plt.plot(bin_centers_plot,w_power_plot,label="Power fit",color="black")

#plt.plot(bin_centers_plot,diff_power_plot,label="data-smooth",color="yellow")

"""

#plotting the bao_angle in the plot
plt.vlines(bao_angle,ymin=np.min(ls_avg_plot),
         ymax=np.max(ls_avg_plot),colors="g")



plt.xlabel("Angle (degrees)")
plt.ylabel("Correlation")
plt.legend()


plt.title(f"{sim_name}")

# --- Save plot as a separate .png file ---
png_filename = filename_histogram.replace('.npz', '_plot.png')
plt.savefig(png_filename)
plt.show()

print(f"Plot saved as '{png_filename}'")

exit()