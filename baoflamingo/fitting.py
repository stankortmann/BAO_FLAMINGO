import warnings
from scipy.optimize import OptimizeWarning
from scipy.optimize import curve_fit
import numpy as np
import unyt as u


class BAO_fitter:
    """
    Class to handle BAO fits: Gaussian peak fitting and monopole/quadrupole template fitting.
    """
    def __init__(self,s_template,
                 mono_template=None, mono_template_err=None,
                 quad_template=None, quad_template_err=None,
                 ):
        """
        Initialize the BAO fitter.

        Parameters
        ----------
        s : np.ndarray
            Array of separations (s) in Mpc
        mono_data : np.ndarray
            Measured monopole correlation function
        mono_data_err : np.ndarray
            Errors on monopole data
        quad_data : np.ndarray, optional
            Measured quadrupole correlation function
        quad_data_err : np.ndarray, optional
            Errors on quadrupole data
        mono_template : np.ndarray, optional
            Theoretical monopole template
        mono_template_err : np.ndarray, optional
            Error on the monopole template (if available)
        quad_template : np.ndarray, optional
            Theoretical quadrupole template
        quad_template_err : np.ndarray, optional
            Error on the quadrupole template (if available)
        """
        
        self.s_template=s_template
        # Monopole
        self.mono_template = mono_template
        self.mono_template_err = mono_template_err

        # Quadrupole (optional)
        self.quad_template = quad_template
        self.quad_template_err = quad_template_err
    # ------------------- Gaussian BAO fit -------------------

    def gaussian(self, s_data, mono_data, mono_data_err,
                init_amplitude=0.005, init_mean=150, init_stddev=5):
        """
        Fit a Gaussian peak to the correlation function with safe handling.
        If the fit fails, returns None for the fit and np.inf for chi2.
        """
        result_dict = {}

        def gauss(x, A, mu, sigma):
            return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                coeff, var_matrix = curve_fit(
                    f=gauss,
                    xdata=s_data,
                    ydata=mono_data,
                    sigma=mono_data_err,
                    p0=(init_amplitude, init_mean, init_stddev)
                )

            std = np.sqrt(np.diag(var_matrix))

            # Relevant coefficients with errors
            result_dict["A"], result_dict["A_err"] = coeff[0], std[0]
            result_dict["mu"], result_dict["mu_err"] = coeff[1], std[1]
            result_dict["sigma"], result_dict["sigma_err"] = coeff[2], std[2]

            result_dict["fit"] = gauss(s_data, *coeff)
            # Optional: compute chi2
            result_dict["chi2"] = np.sum(((mono_data - result_dict["fit"]) / mono_data_err) ** 2)

        except (RuntimeError, ValueError):
            # Fit did not converge
            result_dict["A"] = None
            result_dict["A_err"] = None
            result_dict["mu"] = None
            result_dict["mu_err"] = None
            result_dict["sigma"] = None
            result_dict["sigma_err"] = None
            result_dict["fit"] = None
            result_dict["chi2"] = np.inf

        return result_dict


    # ------------------- Template fit -------------------
    def template_with_shift(self, s_data, mono_data, mono_data_err,
                            include_nuissance=True, poly_order=2):
        """
        Fit the monopole with α free (standard BAO fit).
        Returns a dict with fit results. If fit fails, fit=None.
        """
        result_dict = {}
        
        # Polynomial nuisance model
        def poly(s, coeffs):
            return sum(c * s**i for i, c in enumerate(coeffs))

        # Model with α free
        def model_func(s, bias, alpha, *poly_coeffs):
            xi_interp = np.interp(s / alpha, self.s_template, self.mono_template)
            nuisance = poly(s, poly_coeffs) if include_nuissance else 0.0
            return (bias**2) * xi_interp + nuisance

        # Initial guess (bias, alpha, a_n)
        p0 = [3.0, 1.0] + [0.0] * (poly_order + 1)

        try:
            coeff, cov = curve_fit(
                model_func,
                s_data,
                mono_data,
                sigma=mono_data_err,
                p0=p0
            )

            fit = model_func(s_data, *coeff)
            result_dict["fit"] = fit
            result_dict["alpha"] = coeff[1]
            result_dict["alpha_err"] = np.sqrt(cov[1, 1])
            result_dict["chi2"] = self._chi2(
                data=mono_data,
                data_err=mono_data_err,
                model=fit
            )

        except (RuntimeError, ValueError):
            # Fit did not converge
            result_dict["fit"] = None
            result_dict["alpha"] = None
            result_dict["alpha_err"] = None
            result_dict["chi2"] = np.inf  # or None

        return result_dict

    def template_no_shift(self, s_data, mono_data, mono_data_err,
                        include_nuissance=True, poly_order=2):
        """
        Fit with α fixed to 1 (no BAO scaling).
        Tests how well the template matches the data.
        Returns a dict with fit results. If fit fails, fit=None.
        """
        result_dict = {}

        # Polynomial nuisance model
        def poly(s, coeffs):
            return sum(c * s**i for i, c in enumerate(coeffs))

        # Model with α fixed → only fit polynomial + bias
        def model_func(s, bias, *poly_coeffs):
            xi_interp = np.interp(s, self.s_template, self.mono_template)
            nuisance = poly(s, poly_coeffs) if include_nuissance else 0.0
            return (bias**2) * xi_interp + nuisance

        # Initial guess: bias + polynomial coefficients
        p0 = [1.0] + [0.0] * (poly_order + 1)

        try:
            coeff, cov = curve_fit(
                model_func,
                s_data,
                mono_data,
                sigma=mono_data_err,
                p0=p0
            )

            fit = model_func(s_data, *coeff)
            result_dict["fit"] = fit
            result_dict["bias"] = coeff[0]
            result_dict["bias_err"] = np.sqrt(cov[0, 0])
            result_dict["chi2"] = self._chi2(
                data=mono_data,
                data_err=mono_data_err,
                model=fit
            )

        except (RuntimeError, ValueError):
            # Fit did not converge
            result_dict["fit"] = None
            result_dict["bias"] = None
            result_dict["bias_err"] = None
            result_dict["chi2"] = np.inf  # or None

        return result_dict


    
    def _chi2(self, data, data_err, model):
        return np.sum(((data - model) / data_err)**2)


