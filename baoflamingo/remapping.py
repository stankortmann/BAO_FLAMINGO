import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

# inputs (same as before)
# xi_data (Ns,Nmu), cov (Ns*Nmu, Ns*Nmu), s (Ns,), mu (Nmu,)
# keep_idx, cov_sub, cov_inv already built as in prior code

# --- BAO-only phenomenological template builder (1D in s) ---
def smooth_poly_inv_s(s, coeffs):
    # coeffs = [c0, c1, c2, ...]  -> polynomial in (1/s): sum_k c_k * (1/s)^k
    basis = np.array([ (1.0/s)**k for k in range(len(coeffs)) ])  # shape (K, Ns)
    return np.dot(coeffs, basis)  # shape (Ns,)

def bao_peak(s, s0=102.5, sigma=8.0, A=0.02):
    # Gaussian peak centered at s0 (Mpc/h). A is relative amplitude (peak height ~ A)
    return A * np.exp(-0.5 * ((s - s0)/sigma)**2)

# Build a 2D template function (s,mu) from 1D s template:
def xi_template_from_params(s_arr, mu_arr, coeffs, s0, sigma, A):
    # s_arr: 1D s grid (Ns,), mu_arr: 1D mu grid (Nmu,)
    xi_s = smooth_poly_inv_s(s_arr, coeffs) * (1.0 + bao_peak(s_arr, s0=s0, sigma=sigma, A=A))
    # replicate along mu: assume BAO shape independent of mu (anisotropy handled by remapping)
    Ss, Mu = np.meshgrid(s_arr, mu_arr, indexing='ij')  # (Ns,Nmu)
    xi2d = np.tile(xi_s[:,None], (1, len(mu_arr)))    # (Ns,Nmu)
    return xi2d

# --- Remap (same as earlier) ---
def remap_s_mu(S, Mu, alpha_perp, alpha_par):
    denom = np.sqrt(Mu**2 * alpha_par**2 + (1.0 - Mu**2) * alpha_perp**2)
    s_new  = S * denom
    mu_new = (Mu * alpha_par) / denom
    return s_new, mu_new

# Helper: evaluate phenomenological template at remapped positions (no external interpolator)
Ss_grid, Mu_grid = np.meshgrid(s, mu, indexing='ij')  # full grid

def model_vector_pheno(params):
    """
    params = [alpha_perp, alpha_par, A_bao, s0, sigma, c0, c1, c2, ...]
    where c* are polynomial coeffs for smooth broadband
    Returns flattened model vector restricted to keep_idx ordering.
    """
    alpha_perp, alpha_par = params[0], params[1]
    A_bao = params[2]
    s0    = params[3]
    sigma = params[4]
    coeffs = params[5:]

    # make 1D template in s
    xi_s = smooth_poly_inv_s(s, coeffs) * (1.0 + bao_peak(s, s0=s0, sigma=sigma, A=A_bao))
    # remap coordinates (we need xi_template evaluated at (s',mu')):
    s_new, mu_new = remap_s_mu(Ss_grid, Mu_grid, alpha_perp, alpha_par)  # both (Ns,Nmu)
    # Interpolate xi_s at s_new (1D interp is enough because xi_s independent of mu)
    xi_s_interp = np.interp(s_new.ravel(), s, xi_s, left=0.0, right=0.0).reshape(Ss_grid.shape)
    xi_mod_flat = xi_s_interp.ravel()  # shape Ns*Nmu
    return xi_mod_flat[keep_idx]

# --- chi2 using full covariance submatrix (cov_inv built earlier) ---
def chi2_pheno(params):
    model_sub = model_vector_pheno(params)
    delta = xi_sub - model_sub
    return float(delta @ cov_inv @ delta)

# --- Minimal fit: choose polynomial order and initial guesses ---
poly_order = 3   # number of coeffs
# params: alpha_perp, alpha_par, A_bao, s0, sigma, poly coeffs (c0..c_{poly_order-1})
p0 = np.zeros(5 + poly_order)
p0[0:2] = 1.0, 1.0           # alphas
p0[2] = 0.02                 # A_bao initial amplitude
p0[3] = 102.5                # s0 initial
p0[4] = 8.0                  # sigma
p0[5:] = [1.0, -0.5, 0.1][:poly_order]  # initial polynomial coeffs (tweak if desired)

# bounds or priors can be enforced in minimize by simple checking, but here we do a quick minimize
res = minimize(chi2_pheno, p0, method='Powell')
print("Fit result (phenomenological):", res.x)
print("chi2:", res.fun)
