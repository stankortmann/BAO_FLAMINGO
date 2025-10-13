import numpy as np 


class stat_tools:
    """Weighted statistics and bootstrap methods for preloaded data."""

    def __init__(self, all_ls, weights=None, random_seed=None):
        """
        Initialize with data and optional weights.

        Parameters
        ----------
        all_ls : array-like, shape (n_slices, n_bins)
            Array of slice histograms or measurements.
        weights : array-like, shape (n_slices,), optional
            Weights for each slice. If None, all slices are equally weighted.
        random_seed : int, optional
            Seed for reproducible bootstrap sampling.
        """
        self.all_ls = np.array(all_ls)
        self.n_slices, self.n_bins = self.all_ls.shape
        self.weights = np.ones(self.n_slices) if weights is None else np.array(weights)
        self.prob = self.weights / np.sum(self.weights)
        self.rng = np.random.default_rng(random_seed)

    def weighted_avg_and_std(self):
        """
        Compute the weighted average and standard deviation across slices.

        Returns
        -------
        avg : array, shape (n_bins,)
        std : array, shape (n_bins,)
        """
        avg = np.average(self.all_ls, axis=0, weights=self.weights)
        variance = np.average((self.all_ls - avg)**2, axis=0, weights=self.weights)
        return avg, np.sqrt(variance)

    def bootstrap_slices(self, n_bootstrap=1000, percentiles=(16, 84)):
        """
        Perform bootstrap resampling across slices.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap realizations.
        percentiles : tuple
            Percentiles to compute for confidence intervals.

        Returns
        -------
        mean_ls : array, shape (n_bins,)
            Weighted mean across slices.
        std_ls : array, shape (n_bins,)
            Standard deviation of the bootstrap means.
        ci_low, ci_high : arrays, shape (n_bins,)
            Percentile confidence intervals of the bootstrap.
        """
        mean_ls = np.average(self.all_ls, axis=0, weights=self.weights)
        bootstrap_means = np.zeros((n_bootstrap, self.n_bins))

        for i in range(n_bootstrap):
            idx = self.rng.choice(self.n_slices, self.n_slices, replace=True, p=self.prob)
            bootstrap_means[i, :] = np.mean(self.all_ls[idx, :], axis=0)

        std_ls = np.std(bootstrap_means, axis=0, ddof=1)
        ci_low, ci_high = np.percentile(bootstrap_means, percentiles, axis=0)

        return mean_ls, std_ls, ci_low, ci_high

