from swiftsimio import load
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import formulae as fm
import scipy.spatial as ss



# Of course, replace this path with your own snapshot should you be using
# custom data.
directory="/cosma8/data/dp004/flamingo/Runs/"
simulation="L2800N5040/HYDRO_FIDUCIAL"
safe_simulation = simulation.replace("/", "_") #for directory purposes
soap_hbt_path= directory+simulation+"/SOAP-HBT/halo_properties_00"
redshift_path=directory+simulation+"/output_list.txt"

#only loading snapshort path to get cosmological parameters
snapshot_path = directory+simulation+"/snapshots/flamingo_0020/flamingo_0020.hdf5"
snapshot = load(snapshot_path)

#redshifts are listed in this file for all the snapshots in the "simulation"
redshift_list=np.loadtxt(redshift_path, skiprows=1)
numbers=[74] #which files to load

# Number of "observations" we are going to take
n_slices = 1

#error in redshift determination
error=0.001

#oversampling factor of the random catalogue created
oversampling=5

#Option to only sample central galaxies
central_filtering=False

#Mass filtration percentile, the higher the higher the mass threshold

#currently doing the stellar mass
mass_percentile=92

#Bootstrapping inputs
n_bootstrap=1e6
low_per=16
high_per=84

#histogram setup
min_distance=80 #for now, subject to change
#maximum distance I am interested in in this project, subject to change
max_distance=250 #Mpc
bins = 80


for a in numbers:
    if a < 10: 
        b="0"
    else:
        b=""
    
    data = load(soap_hbt_path+b+str(int(a))+".hdf5")
    print("this is file",str(int(a)), "loaded")
    d_coords=data.input_halos.halo_centre.value
    
    #loading all relevant cosmological parameters fromt snapshot
    metadata=snapshot.metadata
    redshift=redshift_list[a]
    print("this snapshot is at redshift z=",str(redshift))
    
    dz=redshift*error
    cosmology=metadata.cosmology
    box_size = metadata.boxsize.value
    box_size_float=float(box_size[0])
    center= np.array([0.5*box_size_float,
                        0.5*box_size_float,
                        0.5*box_size_float])

    

    #---random catalogue seed, to decrease noise and use same randoms

    r_catalogue_seed=100



    #comoving distance in Mpc:
    comoving_distance,bao_distance=fm.comoving_distance(z=redshift,
                      H0=cosmology.H0.value,
                      Omega_m=cosmology.Om0,
                      Omega_lambda=cosmology.Ode0,
                      Tcmb=cosmology.Tcmb0.value,
                      Neff=cosmology.Neff,
                      Omega_b=cosmology.Ob0)
    
    #calculating comoving distance plus and minus the error so we can filter galaxies
    # based on if their distance is within this redshift bin
    plus_dr,_=fm.comoving_distance(z=redshift+dz,
                      H0=cosmology.H0.value,
                      Omega_m=cosmology.Om0,
                      Omega_lambda=cosmology.Ode0,
                      Tcmb=cosmology.Tcmb0.value,
                      Neff=cosmology.Neff,
                      Omega_b=cosmology.Ob0)
    print(plus_dr)
    
    minus_dr,_=fm.comoving_distance(z=redshift-dz,
                      H0=cosmology.H0.value,
                      Omega_m=cosmology.Om0,
                      Omega_lambda=cosmology.Ode0,
                      Tcmb=cosmology.Tcmb0.value,
                      Neff=cosmology.Neff,
                      Omega_b=cosmology.Ob0)
    

    #calculating the observer parameters to ultimately vary its positions
    #in respect to the box to get multiple observations of the same snapshot
    thickness_slice=plus_dr-minus_dr
    print("The slice thickness is",thickness_slice,"Mpc")
    
    #These are the limits of the observers positions to have the same volume element between
    #these positions


    #In this case we can take a complete spherical cut
    if plus_dr < 0.5*box_size_float:
        
        complete_sphere=True
        
        observer_shift=np.random.uniform(low=-0.5*box_size_float,high=0.5*box_size_float,
                            size=(n_slices,3))
        

        print("In this snapshot a complete spherical slice is possible!!")
       

        



    else:
        #no complete shell can be taken
        complete_sphere=False
        #just to be sure to be inside the box, no rounding down allowed!
        min_x=np.ceil(plus_dr) 
        max_angle_plus_dr=np.arcsin(box_size_float/(2*plus_dr))
        max_x=box_size_float+minus_dr*np.cos(max_angle_plus_dr)-1

        print("In this snapshot NO complete spherical slice is possible")


    # --- FILTERING --- 



    # --- Mass filtration ----

    #only including dark matter and gas
    total_mass=data.exclusive_sphere_500kpc.gas_mass+\
    data.exclusive_sphere_500kpc.dark_matter_mass

    stellar_mass=data.inclusive_sphere_500kpc.stellar_mass

    mass_cut_off=np.percentile(stellar_mass,mass_percentile)
    print("The mass cut-off for the",mass_percentile,"mass percentile is",mass_cut_off)
    mass_mask= stellar_mass >= mass_cut_off


    #release later
    d_coords= d_coords[mass_mask]

    
    
    # --- Central filtration  ---
        
    if central_filtering==True:
        d_coords= fm.galaxy_type(
            coordinates=d_coords,
            data=data,
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
        # Convert to spherical wrt this observer
        if complete_sphere == False:
            #always recenter the observer
            observer = center
            x_y_z = np.random.randint(3)
            observer[x_y_z]=np.random.uniform(low=min_x,
                                            high=max_x)
            d_coords_sph = fm.cartesian_to_spherical(d_coords, observer=observer,
                                box_size=box_size_float,sphere=complete_sphere)
        
        if complete_sphere == True:

            d_coords_sph = fm.cartesian_to_spherical(d_coords-observer_shift[i],
                                 observer=center,
                                box_size=box_size_float,sphere=complete_sphere)
            

       
        
       

        # Apply redshift shell filtering
        r_mask = (d_coords_sph[:, 0] < plus_dr) & (d_coords_sph[:, 0] > minus_dr)
        d_coords_filtered = d_coords_sph[r_mask][:, 1:]  # keep theta, phi
        
        #d_coords_filtered=d_coords_filtered[:20000]

        #how many D points in the spherical slice do we have
        data_size = np.shape(d_coords_filtered)[0]
        
        if data_size < 2:
            print("Empty slice",i,"after redshift filtering.")
            continue  # skip empty slice, we know we did something wrong!

        
            
        #--What is the data_size?
        print("The data size in slice",str(i+1),"is",data_size)

        #-- Determining the random catalogue size, only the first time
        if i==0:
            #number of randoms (oversampling)
            n_randoms = int(oversampling* data_size)

        # --- Compute pair counts and correlation ---

        #if distance_type='r' , the distances are in Mpc
        #if distance_type='theta', the distance are in degrees
        # I want to calculate rr right now, only once in the first slice
        if i==0:
            dd, dr, rr = fm.angular_separations_chord(
                data=d_coords_filtered,
                box_size=box_size_float,
                radius=comoving_distance,
                max_distance=max_distance,     #All lengths in Mpc
                bao_distance=bao_distance,
                complete_sphere=complete_sphere,
                seed=r_catalogue_seed,
                n_randoms=n_randoms,
                slice_index=i,
                distance_type='r'      
            )
        if i !=0:
            dd, dr,_ = fm.angular_separations_chord(
                data=d_coords_filtered,
                box_size=box_size_float,
                radius=comoving_distance,
                max_distance=max_distance,     #All lengths in Mpc
                bao_distance=bao_distance,
                complete_sphere=complete_sphere,
                seed=r_catalogue_seed,
                n_randoms=n_randoms,
                slice_index=i,
                distance_type='r'       
            )

        #only once again
        if i ==0:
            # Histogram setup, same for all data outputs
            
            
            
            bin_array = np.linspace(min_distance, max_distance, bins+1)

            hist_rr, bin_edges = np.histogram(rr, bins=bin_array)

            rr_norm = hist_rr / (n_randoms * (n_randoms - 1) / 2)




            

        hist_dd, _ = np.histogram(dd, bins=bin_array)
        hist_dr, _ = np.histogram(dr, bins=bin_array)
        
        
        #normalization depending on the sample size
        dd_norm = hist_dd / (data_size * (data_size - 1) / 2)
        dr_norm = hist_dr / (data_size * n_randoms)

        
            
            

        hist_ls = (dd_norm - 2*dr_norm + rr_norm) / rr_norm

        

        all_ls.append(hist_ls) #appending all the hists for every slice
        all_data_size.append(data_size)
        print("slice number",str(i+1),"is done!!!")

    
    print("All the slices are done!")

    bin_centers = 0.5 * (bin_array[:-1] + bin_array[1:])
    bin_width = np.diff(bin_edges)[0]  # width of each bin

    # --- Save histogram to text file ---
    output_filename = "pdh_avgs_"+safe_simulation+"_data_00"+b+str(int(a))+".npz"
    

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

            min_distance=min_distance,
            max_distance=max_distance,
            

            bao_distance=bao_distance,
            

            #normal statistics
            ls_avg=all_ls)

    




        

    print("Saved averaged Landy-Szalay histograms with std across slices.")
    
    print("Data saved in",output_filename)


    
    






exit()