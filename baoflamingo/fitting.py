import numpy as np
from scipy.optimize import curve_fit


def gaussian_data(distances, 
                correlation, 
                yerror,
                initial_amplitude=0.005, 
                initial_mean=150, 
                initial_stddev=5):
    # Define model function to be used to fit to the data above:
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    coeff, var_matrix = curve_fit(gauss, 
                                distances, 
                                correlation,
                                sigma=yerror,
                                p0=(initial_amplitude, initial_mean, initial_stddev))

    # Get the fitted curve
    hist_fit = gauss(distances, *coeff)
    perr = np.sqrt(np.diag(var_matrix))


    # Finally, lets get the fitting parameters, i.e. the mean and standard deviation with standard deviations:
    mu=(coeff[1],perr[1])
    sigma=(coeff[2],perr[2])
    

    return hist_fit,mu,sigma


