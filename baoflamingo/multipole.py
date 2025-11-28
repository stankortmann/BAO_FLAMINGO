import numpy as np
from numpy.polynomial.legendre import legval


class multipole_projector:
    """
    Takes (s, mu) binning and ell list, builds projection matrix P,
    and projects a full 2D covariance to multipoles with errors.
    """

    def __init__(self,xi, cov, mu, s, ell_list=(0, 2), mu_range_minus1_to1=True,regularize=None):
        
        self.xi = np.asarray(xi)
        self.cov = np.asarray(cov)
        self.mu = np.asarray(mu)
        self.s = np.asarray(s)
        self.Ns = len(s)
        self.Nmu = len(mu)
        self.ell_list = ell_list
        self.mu_range_minus1_to1 = mu_range_minus1_to1
        self.regularize = regularize
        self.dmu = self._compute_dmu(self.mu)

        #running the pipeline inside the class init function

        self._build_P() #build projection matrix P of all the multipoles you have selected 
        self._project_covariance() #project the full covariance to multipole covariance
        self._get_multipoles_with_errors()#actually compute the multipoles and their errors (1 sigma)  

    

    
    def _compute_dmu(self):
        """Compute per-bin Δμ allowing for non-uniform μ spacing."""
        
        if self.mu.size == 1:
            raise ValueError("Cannot infer dmu with a single μ bin.")

        dmu = np.empty_like(mu)
        dmu[0] = self.mu[1] - self.mu[0]
        dmu[-1] = self.mu[-1] - self.mu[-2]

        if mu.size > 2:
            dmu[1:-1] = 0.5 * (mu[2:] - mu[:-2])
        
        self.dmu = dmu

        

    # ----------------------------------------------------------------------

    def _build_P(self):
        """Build the projection matrix P once and store it."""
        

        Nout = len(self.ell_list) * self.Ns
        Nin = self.Ns * self.Nmu

        P = np.zeros((Nout, Nin), dtype=float)

        for i_s in range(Ns):
            col_start = i_s * Nmu

            for ell_idx, ell in enumerate(self.ell_list):

                row = ell_idx * Ns + i_s

                if ell == 0:
                    # monopole weights
                    if self.mu_range_minus1_to1:
                        w = 0.5 * self.dmu
                    else:
                        w = self.dmu

                elif ell == 2:
                    P2 = 0.5 * (3.0 * mu**2 - 1.0)
                    if self.mu_range_minus1_to1:
                        w = 0.5 * 5.0 * P2 * self.dmu
                    else:
                        w = 5.0 * P2 * self.dmu

                else:
                    Pl = legval(mu, [0]*ell + [1])
                    if self.mu_range_minus1_to1:
                        w = 0.5 * (2*ell + 1) * Pl * dmu
                    else:
                        w = (2*ell + 1) * Pl * dmu

                P[row, col_start:col_start + Nmu] = w

        self.P = P

    # ----------------------------------------------------------------------
    # Main functionality
    # ----------------------------------------------------------------------

    def _project_covariance(self):
        """
        Project full covariance (Ns*Nmu)^2 → multipole covariance.

        Returns:
            C_multi   - projected multipole covariance
            xi_errs   - dict of per-multipole errors
            blocks    - sub-blocks (xi0, xi2, and cross terms)
            P         - projection matrix
        """
        
        assert self.cov.shape == (self.Ns*self.Nmu, self.Ns*self.Nmu), "2D covariance shape mismatch"

        C_full = self.cov.copy()

        # Optional covariance regularization, might look into this more later!!!
        if self.regularize is not None:
            eps = regularize * np.trace(C_full) / C_full.shape[0]
            C_full += eps * np.eye(C_full.shape[0])

        # project
        
        C_multi = self.P @ C_full @ self.P.T
        #to ensure symmetry
        C_multi = 0.5 * (C_multi + C_multi.T)

        # extract errors and block structure
        xi_errs = {}
        blocks = {}

        for idx, ell in enumerate(self.ell_list):
            start = idx * Ns
            stop = (idx + 1) * Ns
            block = C_multi[start:stop, start:stop]

            xi_errs[ell] = np.sqrt(np.clip(np.diag(block), 0.0, None))
            blocks[f'ell{ell}_cov'] = block

        # cross-blocks
        if len(self.ell_list) > 1:
            for i, ell_i in enumerate(self.ell_list):
                for j, ell_j in enumerate(self.ell_list):
                    if j <= i:
                        continue
                    si, ei = i*Ns, (i+1)*Ns
                    sj, ej = j*Ns, (j+1)*Ns
                    blocks[f'ell{ell_i}_ell{ell_j}_cov'] = C_multi[si:ei, sj:ej]
                    blocks[f'ell{ell_j}_ell{ell_i}_cov'] = blocks[f'ell{ell_i}_ell{ell_j}_cov'].T

        self.C_multi, self.xi_errs, self.blocks, self.P = C_multi, xi_errs, blocks, P

    def _get_multipoles_with_errors(self):
        """
        Compute the multipoles (ell in self.ell_list) and their 1σ errors.
        Returns:
            multipoles : dict { ell : xi_ell(s) }
            errors     : dict { ell : sigma_ell(s) }
            stacked    : 1D array stacking all multipoles [xi_ell1, xi_ell2, ...]
            stacked_err: 1D array stacking corresponding errors
        Notes:
            - Ignores cross-covariances between different ℓ.
            - Uses only the diagonal of the multipole covariance blocks.
        """

        xi=self.xi.copy()  # shape: (Ns, Nmu)
        xi_flat = xi.reshape(self.Ns * self.Nmu)

       #covariance and p are already built in the init function

        # ----- Project the multipoles themselves -----
        xi_multi = self.P @ xi_flat   # shape: (len(ell_list)*Ns,)

        # ----- Extract each multipole into a dictionary -----
        multipoles = {}
        errors = {}

        for idx, ell in enumerate(self.ell_list):
            start = idx * self.Ns
            stop  = (idx + 1) * self.Ns

            # xi_ell(s)
            xi_ell = xi_multi[start:stop]

            # σ_ell(s) from diagonal blocks
            sigma_ell = xi_errs[ell]

            multipoles[ell] = xi_ell
            errors[ell] = sigma_ell


        self.multipoles = multipoles
        self.errors = errors







    
