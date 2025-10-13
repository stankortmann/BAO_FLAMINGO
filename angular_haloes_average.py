from swiftsimio import load
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as ss

#now our own modules
import galaxy_correlation as gal_cor
import filtering as flt



# Of course, replace this path with your own snapshot should you be using
# custom data.
directory="/cosma8/data/dp004/flamingo/Runs/"
simulation="L2800N5040/HYDRO_FIDUCIAL"
safe_simulation = simulation.replace("/", "_") #for directory purposes
soap_hbt_path= directory+simulation+"/SOAP-HBT/halo_properties_00"
redshift_path=directory+simulation+"/output_list.txt"

#only loading snapshort path to get cosmological parameters
snapshot_path = directory+simulation+"/snapshots/flamingo_0020/flamingo_0020.hdf5"
#snapshot = load(snapshot_path)

#redshifts are listed in this file for all the snapshots in the "simulation"
redshift_list=np.loadtxt(redshift_path, skiprows=1)
numbers=[72] #which files to load

# Number of "observations" we are going to take
n_slices = 20

#maximum distance I am interested in in this project, subject to change
min_distance=0 #Mpc
max_distance=300 #Mpc

#error in redshift determination
#We will have to make this dependent on redshift later
error=0.001

#oversampling factor of the random catalogue created and rng seed
seed_random=12345
oversampling=5

#Option to only sample central galaxies
central_filtering=False

#Mass filtration percentile, higher = higher mass threshold
mass_percentile=90

#Bootstrapping inputs
n_bootstrap=1e6
low_per=16
high_per=84

#histogram setup
bins = 200


for a in numbers:
    if a < 10: 
        b="0"
    else:
        b=""
    
    data = load(soap_hbt_path+b+str(int(a))+".hdf5")
    print("this is file",str(int(a)), "loaded")
    d_coords=data.input_halos.halo_centre.value
    
    #loading all relevant cosmological parameters fromt snapshot
    metadata=soap.metadata
    redshift=redshift_list[a]
    print("this snapshot is at redshift z=",str(redshift))
    
    dz=redshift*error
    cosmology=metadata.cosmology
    box_size = metadata.boxsize.value
    box_size_float=float(box_size[0])
    middle=box_size_float/2
    centre=np.array([middle,middle,middle])

    #---random catalogue seed, to decrease noise and use same randoms

    r_catalogue_seed=np.random.randint(500)

    #---Cosmology tools object
    cosmo_tools=gal_cor.cosmo_tools(
        H0=cosmology.H0.value,
        Omega_m=cosmology.Om0,
        Omega_b=cosmology.Ob0,
        Omega_lambda=cosmology.Ode0,
        Tcmb=cosmology.Tcmb0.value,
        Neff=cosmology.Neff
    )

    #comoving distance in Mpc:
    comoving_distance=gal_cor.cosmo_tools.comoving_distance(z=redshift)
    print("The comoving distance to redshift z is",comoving_distance,"Mpc")

    angular_diameter_distance=gal_cor.cosmo_tools.angular_diameter_distance(z=redshift)

    luminosity_distance=gal_cor.cosmo_tools.luminosity_distance(z=redshift)

    #BAO scale in Mpc
    bao_distance=gal_cor.cosmo_tools.bao_sound_horizon
    print("The BAO sound horizon is",bao_distance,"Mpc")
    
    #calculating comoving distance plus and minus the error so we can filter galaxies
    # based on if their distance is within this redshift bin
    plus_dr=gal_cor.cosmo_tools.comoving_distance(z=redshift+dz)
    
    minus_dr=gal_cor.cosmo_tools.comoving_distance(z=redshift-dz)
    

    #calculating the observer parameters to ultimately vary its positions
    #in respect to the box to get multiple observations of the same snapshot
    thickness_slice=plus_dr-minus_dr
    print("The slice thickness is",thickness_slice,"Mpc")
    
    #These are the limits of the observers positions to have the same volume element between
    #these positions



    #In this case we can take a complete spherical cut
    if plus_dr < 0.5*box_size_float:
        
        complete_sphere=True
        # we can place the observer anywhere in the box
        # We randomly shift the coordinates periodically to get different observations
        # in the box
        #The observer is always at the centre of the box
        shift_coordinates=np.random.uniform(
            low=-0.5*box_size_float,high=0.5*box_size_float,
                            size=(n_slices,3))
        

        print("In this snapshot a complete spherical slice is possible!!")

        



    else:
        #no complete shell can be taken
        complete_sphere=False
        #just to be sure to be inside the box, no rounding down allowed!
        min_x=np.ceil(plus_dr) 
        max_angle_plus_dr=np.arcsin(box_size_float/(2*plus_dr))
        max_x=box_size_float+minus_dr*np.cos(max_angle_plus_dr)
        
        shift_observer_xyz=np.random.uniform(
            low=min_x,high=max_x,
            size=n_slices)
        #Which coordinate to shift
        x_y_z_list = np.random.randint(3,size=n_slices)                
        

        print("In this snapshot NO complete spherical slice is possible")





    # --- FILTERING --- 



    # --- Mass filtration ----

    #only including dark matter and gas
    total_mass=data.spherical_overdensity_200_crit.gas_mass+\
    data.spherical_overdensity_200_crit.dark_matter_mass

    mass_cut_off=np.percentile(total_mass,mass_percentile)
    print("The mass cut-off for the",mass_percentile,"mass percentile is",mass_cut_off)
    mass_mask= total_mass >= mass_cut_off

    d_coords= d_coords[mass_mask]

    
    
    # --- Central filtration  ---
        
    if central_filtering==True:
        d_coords= fm.galaxy_type(
            data=d_coords,
            sort="central"
        )
            

    #we can later implement multiple slices of the same simulation
    #x_observer>plus_dr and <box_size+minus_dr

    #No we want to place an observer at a distance of comoving_distance from the
    #centre of the box in the x direction
    #so the observer is at (500+comoving_distance, 500, 500)
    
    

    all_ls = []  # store Landy-Szalay histograms
    all_data_size = [] # store data sizes of each slice

    for i in range(n_slices):
        
        
        # --- Observer position ---
        
        
        if complete_sphere:
            #observer always in the centre
            observer=centre.copy()
            #shifting of coordinates
            shift=shift_coordinates[i]
            

        else:
            
            #always again
            observer = centre.copy()
            #shifting the observer in one of the coordinates
            observer[x_y_z_list] = shift_observer_xyz[i]
            #no shifting of coordinates
            shift=np.array([0,0,0])
            

        #Maybe the next lines can be done faster but I have to think about this later!!!
        
        #Initialize the coordinate transformation class
        coordinates = gal_cor.coordinate_tools(
            coordinates=d_coords,box_size=box_size_float,
        complete_sphere=complete_sphere,observer=None,shift=None)

        #optimized with numba
        d_coords_sph = coordinates.spherical



        ###########################################
        #apply filtering to the data coordinates
        ###########################################










        # Apply redshift shell filtering
        r_mask = (d_coords_sph[:, 0] < plus_dr) & (d_coords_sph[:, 0] > minus_dr)
        d_coords_filtered = d_coords_sph[r_mask][:, 1:]  # keep theta, phi

        # Size of the filtered data aka number of galaxies in the slice
        data_size = np.shape(d_coords_filtered)[0]
        
        if data_size < 2:
            print("Empty slice",i,"after redshift filtering.")
            continue  # skip empty slice, we know we did something wrong!

        
            
        #--What is the data_size?
        print("The data size in slice",str(i+1),"is",data_size)

        #-- Determining the random catalogue size, only the first time
        if i==0:
            #number of randoms (oversampling)
            n_random = int(oversampling*data_size)



        #### Initializing the correlation tools class
        #Might ultimately be done outside the loop, we have to know the data size first
        #to determine the random catalogue size
        correlation=gal_cor.correlation_tools(
            box_size=1000.0, radius=427.0, max_angle_plus_dr=10.0, 
            min_distance=min_distance , max_distance=max_distance, bao_distance=bao_distance,
            complete_sphere=complete_sphere, 
            seed=seed_random, n_random=n_random,
            bins=bins, distance_type='angular')

        hist_ls=correlation.landy_szalay(
            data_coords=d_coords_filtered)

        all_ls.append(hist_ls) #appending all the hists for every slice
        all_data_size.append(data_size)
        print("slice number",str(i+1),"is done!!!")

    
    print("All the slices are done!")
    
    # --- Average and standard deviation across slices ---
    all_ls = np.array(all_ls)  # shape (n_slices, n_bins)
    
    all_data_size=np.array(all_data_size) #shape (n_slice)
    
    mean_ls,std_ls=fm.weighted_avg_and_std(
        values=all_ls, weights=None)
    mean_ls_weighted,std_ls_weighted=fm.weighted_avg_and_std(
        values=all_ls, weights=all_data_size)

    bin_centers = correlation.bin_centers
    bin_width = correlation.bin_width







###################################################################














    # --- BOOTSTRAPPING  ----
    
    print("We now start bootstrapping!")

    ls_avg_bs, ls_std_bs, ci_low_bs, ci_high_bs = fm.bootstrap_slices(
    all_ls=all_ls,
    weights=all_data_size,
    n_bootstrap=int(n_bootstrap),
    percentiles=(low_per, high_per),
    random_seed=np.random.randint(low=0,high=int(1e7))
) 

    print("We are done bootstrapping")

    # --- Save histogram to text file ---
    output_filename = "pdh_angular_avgs_"+safe_simulation+"_data_00"+b+str(int(a))+".npz"


    # Save to file
    np.savez(
            output_filename,
            bin_centers=bin_centers,
            bin_width=bin_width,
            bins=bins,
            slices=n_slices,
            thickness_slice=thickness_slice,
            
            data_sizes=all_data_size,
            oversampling=oversampling,
            random_size=n_randoms,
            central_filtering=central_filtering,
            mass_cut_off=mass_cut_off,

            min_angle=min_angle,
            max_angle=max_angle,
            bao_angle=bao_angle,

            bao_distance=bao_distance,
            max_distance=max_distance,

            #normal statistics
            ls_avg=mean_ls,
            ls_std=std_ls,
            #weighted statistics slice sizes
            ls_avg_weighted=mean_ls_weighted,
            ls_std_weighted=std_ls_weighted,

            #bootstrap statistics
            ls_avg_bs=ls_avg_bs,
            ls_std_bs=ls_std_bs,
            
            n_bootstrap=n_bootstrap,
            ci_low_bs=ci_low_bs,
            ci_high_bs=ci_high_bs,
            low_per=low_per,
            high_per=high_per




        )

    print("Saved averaged Landy-Szalay histograms with std across slices.")
    
    print("Data saved in",output_filename)


    
    






exit()