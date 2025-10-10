import numpy as np 




class stat_tools:
    """Weighted statistics and bootstrap methods."""

    @staticmethod
    def weighted_avg_and_std(values, weights=None):
        avg = np.average(values, axis=0, weights=weights)
        variance = np.average((values-avg)**2, axis=0, weights=weights)
        return avg, np.sqrt(variance)

    @staticmethod
    def bootstrap_slices(all_ls, weights=None, n_bootstrap=1000, percentiles=(16,84), random_seed=None):
        all_ls = np.array(all_ls)
        n_slices, n_bins = all_ls.shape
        rng = np.random.default_rng(random_seed)
        weights = np.ones(n_slices) if weights is None else np.array(weights)
        prob = weights / np.sum(weights)
        mean_ls = np.average(all_ls, axis=0, weights=weights)
        bootstrap_means = np.zeros((n_bootstrap, n_bins))
        for i in range(n_bootstrap):
            idx = rng.choice(n_slices, n_slices, replace=True, p=prob)
            bootstrap_means[i,:] = np.mean(all_ls[idx,:], axis=0)
        std_ls = np.std(bootstrap_means, axis=0, ddof=1)
        ci_low, ci_high = np.percentile(bootstrap_means, percentiles, axis=0)
        return mean_ls, std_ls, ci_low, ci_high
