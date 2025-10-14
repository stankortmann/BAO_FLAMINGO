from swiftsimio import load
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as ss
import psutil
import os
import time
import gc

#now our own modules
import galaxy_correlation as gal_cor
import filtering as flt
import statistics as stat
import threading




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




# Of course, replace this path with your own snapshot should you be using
# custom data.
directory="/cosma8/data/dp004/flamingo/Runs/"
simulation="L2800N5040/HYDRO_FIDUCIAL"
safe_simulation = simulation.replace("/", "_") #for directory purposes
soap_hbt_path= directory+simulation+"/SOAP-HBT/halo_properties_00"
redshift_path=directory+simulation+"/output_list.txt"


#redshifts are listed in this file for all the snapshots in the "simulation"
redshift_list=np.loadtxt(redshift_path, skiprows=1)
numbers=[72] #which files to load

# Number of "observations" we are going to take
n_slices = 20

#maximum distance I am interested in in this project, subject to change
min_distance=0 #Mpc
max_distance=220 #Mpc

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

#luminosity apparent magnitude cutoff
mr=19


for a in numbers:
    if a < 10: 
        b="0"
    else:
        b=""
    
    data = load(soap_hbt_path+b+str(int(a))+".hdf5")
    print("this is file",str(int(a)), "loaded")
    
    
    #loading all relevant cosmological parameters from the metadata
    metadata=data.metadata
    redshift=redshift_list[a]
    print("this snapshot is at redshift z=",str(redshift))
    cosmology=metadata.cosmology
    box_size = metadata.boxsize.value
    box_size_float=float(box_size[0])
    middle=box_size_float/2
    centre=np.array([middle,middle,middle])

    #---random catalogue seed, to decrease noise and use same randoms

    r_catalogue_seed=np.random.randint(500)

    #---Cosmology tools object
    cosmo=gal_cor.cosmo_tools(
        H0=cosmology.H0.value,
        Omega_m=cosmology.Om0,
        Omega_b=cosmology.Ob0,
        Omega_lambda=cosmology.Ode0,
        Tcmb=cosmology.Tcmb0.value,
        Neff=cosmology.Neff,
        redshift=redshift,
        n_sigma=2 #3 sigma redshift bin

    )
    print("The cosmology is set up, now metadata file can be closed")
    del metadata
    gc.collect()


    #comoving distance in Mpc:
    comoving_distance=cosmo.comoving_distance
    print("The comoving distance to redshift z is",comoving_distance,"Mpc")


    
    #calculating comoving distance plus and minus the error so we can filter galaxies
    # based on if their distance is within this redshift bin

    

    #calculating the observer parameters to ultimately vary its positions
    #in respect to the box to get multiple observations of the same snapshot
    
    print("The slice thickness is",cosmo.delta_dr,"Mpc")
    
    #These are the limits of the observers positions to have the same volume element between
    #these positions



    #In this case we can take a complete spherical cut
    if cosmo.plus_dr < 0.5*box_size_float:
        
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

        #extra attention here, this is not yet comletely correct
        min_x=np.ceil(cosmo.plus_dr) 
        max_angle_plus_dr=np.arcsin(box_size_float/(2*cosmo.plus_dr))
        max_x=box_size_float+cosmo.minus_dr*np.cos(max_angle_plus_dr)
        
        shift_observer_xyz=np.random.uniform(
            low=min_x,high=max_x,
            size=n_slices)
        #Which coordinate to shift
        x_y_z_list = np.random.randint(3,size=n_slices)                
        

        print("In this snapshot NO complete spherical slice is possible")





    # --- FILTERING --- 
    filters=flt.filtering_tools(
        soap_file=data,
        cosmology=cosmo,    #send the instance of the cosmology class
        central_filter=False, 
        stellar_mass_filter=False,stellar_mass_cutoff=0,
        luminosity_filter=True, filter_band='r' ,m_cutoff=mr #luminosity filter parameters
        )
    gc.collect()
   
    print("The filters are set up and will now be applied to the data")
    
    d_coords=data.input_halos.halo_centre.value
    luminosity_mask=filters.luminosity_filter()
    gc.collect()
    d_coords=d_coords[luminosity_mask] #apply the total mask to the coordinates
    print("The number of galaxies after filtering is",np.shape(d_coords)[0])

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
            coordinates=d_coords,
            box_size=box_size_float,
            complete_sphere=complete_sphere,
            observer=observer,
            shift=shift
            )

        #optimized with numba
        d_coords_sph = coordinates.spherical
        

        # Apply redshift shell filtering and overwrite d_coords_sph for memory efficiency
         
        spherical_mask = filters.radial_filter(d_coords_sph)
        d_coords_sph=d_coords_sph[spherical_mask] [:,1:]# keep theta, phi
         
        
        # Size of the filtered data aka number of galaxies in the slice
        data_size = np.shape(d_coords_sph)[0]
        
        if data_size < 2:
            print("Empty slice",i,"after redshift filtering.")
            continue  # skip empty slice, we know we did something wrong!

        
            
        #--What is the data_size?
        print("The data size in slice",str(i+1),"is",data_size)

        #-- Determining the random catalogue size and correlation class
        #only the first time so it is faster!
        if i==0:
            #number of randoms (oversampling)
            n_random = int(oversampling*data_size)



            #### Initializing the correlation tools class
            #Might ultimately be done outside the loop, we have to know the data size first
            #to determine the random catalogue size
            correlation=gal_cor.correlation_tools(
                box_size=1000.0, radius=427.0, max_angle_plus_dr=10.0, 
                min_distance=min_distance , max_distance=max_distance,
                bao_distance=cosmo.bao_distance,
                complete_sphere=complete_sphere, 
                seed=seed_random, n_random=n_random,
                bins=bins, distance_type='angles')

        hist_ls=correlation.landy_szalay(
            data_coords=d_coords_sph)

        all_ls.append(hist_ls) #appending all the hists for every slice

        #Calculate the survey density in galaxies per square degree
        survey_density=correlation.galaxy_density(data_size)

        all_data_size.append(data_size)
        print("slice number",str(i+1),"is done!!!")

    
    print("All the slices are done!")
    
    # --- STATISTICS ---
    all_ls=np.array(all_ls)
    all_data_size=np.array(all_data_size)
    stats = stat.stat_tools(
        all_ls=all_ls, 
        weights=all_data_size, 
        random_seed=12345
        )

    
    #standard weighted statistics
    mean_ls, std_ls = stats.weighted_avg_and_std()


    #bootstrappign
    mean_bs, std_bs, ci_low, ci_high = stats.bootstrap_slices(
        n_bootstrap=n_bootstrap,
        percentiles=percentiles
        )




    

    # --- Save histogram to text file ---
    output_filename = "pdh_angular_avgs_"+safe_simulation+"_data_00"+b+str(int(a))+".npz"


    # Save to file
    np.savez(
            output_filename,
            bin_centers=correlation.bin_centers,
            bin_width=correlation.bin_width,
            bins=correlation.bins,
            slices=n_slices,
            thickness_slice=cosmo.delta_dr,
            
            data_sizes=all_data_size,
            oversampling=oversampling,
            random_size=correlation.n_randoms,
            

            min_angle=correlation.min_angle,
            max_angle=correlation.max_angle,
            bao_angle=correlation.bao_angle,

            bao_distance=cosmo.bao_distance,
            max_distance=max_distance,

            #normal statistics with weights
            ls_avg=mean_ls,
            ls_std=std_ls,
            

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