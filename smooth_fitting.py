def fit_smooth_correlation(theta, w_theta, theta_min=None, theta_max=None):
    """
    Fit a smooth power-law correlation function w(theta) = A * theta^(-gamma)
    to measured angular correlation data.

    Parameters
    ----------
    theta : array-like
        Angular separations (degrees).
    w_theta : array-like
        Measured angular correlation function values.
    theta_min : float, optional
        Minimum theta to use for fitting (exclude large-scale BAO bump).
    theta_max : float, optional
        Maximum theta to use for fitting (exclude large-scale BAO bump).

    Returns
    -------
    w_smooth : np.ndarray
        Fitted smooth correlation function evaluated at input theta.
    A_fit : float
        Fitted amplitude of the power law.
    gamma_fit : float
        Fitted slope of the power law.
    """
    # Power-law model
    def power_law(t, A, gamma):
        return A * t**(-gamma)

    
    #polynomial
    def polynomial(theta, a0, a1, a2):
        """
        Smooth angular correlation function (no BAO bump).
        Quadratic polynomial in log(theta).
        
        Parameters
        ----------
        theta : array-like
            Angular separation (in degrees or radians, but consistent!)
        a0, a1, a2 : floats
            Polynomial coefficients
        
        Returns
        -------
        w : array-like
            Smooth model correlation
        """
        logt = np.log(theta)
        return a0 + a1*logt + a2*logt**2


    theta = np.array(theta)
    w_theta = np.array(w_theta)

    # Mask theta range if specified
    mask = np.ones_like(theta, dtype=bool)
    if theta_min is not None:
        mask &= (theta >= theta_min)
    if theta_max is not None:
        mask &= (theta <= theta_max)

    theta_fit = theta[mask]
    w_fit = w_theta[mask]

    # Fit power-law
    popt_power, _ = curve_fit(power_law, theta_fit, w_fit, p0=[0.01, 1.7])
    A_fit, gamma_fit = popt_power
    w_power = power_law(theta,*popt_power)

    # Fit polynomial
    popt_poly, _ = curve_fit(polynomial,theta_fit,w_fit,p0=[0.1,-0.1,0.1])
    a0_fit, a1_fit, a2_fit = popt_poly
    w_poly = polynomial(theta, *popt_poly)

    return w_power, w_poly