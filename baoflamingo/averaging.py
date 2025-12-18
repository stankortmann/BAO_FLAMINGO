import os
import h5py
import numpy as np
from glob import glob
from collections import defaultdict
import unyt as u

class xi_cov_averager:
    """
    Automatically load and average correlation function outputs across run_xxx folders.
    """

    def __init__(self, base_dir, use_volume_density_weights=True):
        self.use_volume_density_weights = use_volume_density_weights
        self.base_dir = base_dir
        self.file_groups = self._collect_file_groups()
        self._run_all()

    # ---------------------------------------------------------
    # FILE COLLECTION: Group by filename across runs
    # ---------------------------------------------------------
    def _collect_file_groups(self):
        pattern = os.path.join(self.base_dir, "run_*", "*.hdf5")
        all_files = sorted(glob(pattern))

        if len(all_files) == 0:
            raise FileNotFoundError(f"No .hdf5 files found under {self.base_dir}/run_*/")

        groups = defaultdict(list)

        for fp in all_files:
            filename = os.path.basename(fp)
            groups[filename].append(fp)

        return dict(groups)

    # ---------------------------------------------------------
    # COPY METADATA FROM FIRST FILE
    # ---------------------------------------------------------
    def _copy_metadata(self, h5file):
        meta = {}
        attrs = {}

        for key in h5file.keys():
            if key in ("xi", "cov"):
                continue
            meta[key] = h5file[key][()]
            attrs[key] = dict(h5file[key].attrs)

        return {"data": meta, "attrs": attrs}

    # ---------------------------------------------------------
    # AVERAGE A SINGLE GROUP OF FILEPATHS
    # ---------------------------------------------------------
    def _process_group(self, filename, filepaths):
        xis, covs = [], []
        vol_fracs, densities = [], []
        meta = None

        for i, fp in enumerate(filepaths):
            with h5py.File(fp, "r") as f:
                xis.append(f["xi"][()])
                covs.append(f["cov"][()])

                vol_fracs.append(f["survey_total_volume_percent"][()] / 100.0)
                densities.append(f["survey_density"][()])

                if i == 0:
                    meta = self._copy_metadata(f)

        xis = np.array(xis)          # (N, Ns, Nmu)
        covs = np.array(covs)        # (N, Ns*Nmu, Ns*Nmu)
        vol_fracs = np.array(vol_fracs)
        densities = np.array(densities)

        # -------------------------------------------------
        # WEIGHTS
        # -------------------------------------------------
        if self.use_volume_density_weights:
            raw_w = vol_fracs * densities
            w = raw_w / np.sum(raw_w)
        else:
            w = np.ones(len(filepaths)) / len(filepaths)

        # -------------------------------------------------
        # AVERAGE xi(s, mu)
        # -------------------------------------------------
        xi_avg = np.tensordot(w, xis, axes=(0, 0))

        # -------------------------------------------------
        # AVERAGE COVARIANCE (constant rho)
        # -------------------------------------------------
        rho = np.mean(vol_fracs)
        alpha = w**2 + rho * w * (1 - w)

        cov_avg = np.zeros_like(covs[0])
        for i in range(len(w)):
            cov_avg += alpha[i] * covs[i]

        # -------------------------------------------------
        # EFFECTIVE STATISTICS
        # -------------------------------------------------
        n_runs_used = len(filepaths)

        # proportional galaxy counts per slice
        N_i = densities * vol_fracs

        # effective volume (fraction of box)
        V_eff = np.sum(w * vol_fracs)

        # Kish effective number of galaxies
        N_eff = (np.sum(w * N_i))**2 / np.sum(w**2 * N_i)

        # effective density
        density_eff = N_eff / V_eff

        stats = {
            "n_runs_used": n_runs_used,
            "effective_volume_fraction": V_eff,
            "effective_n_galaxies": N_eff,
            "effective_density": density_eff,
        }

        return xi_avg, cov_avg, meta, stats


    # ---------------------------------------------------------
    # WRITE AVERAGED OUTPUT
    # ---------------------------------------------------------
    def _write_output(self, filename, xi_avg, cov_avg, meta, stats):
        """
        Write averaged xi, covariance, metadata, and automatically all stats (with units if provided).
        """
        outdir = os.path.join(self.base_dir, "average")
        os.makedirs(outdir, exist_ok=True)

        outpath = os.path.join(outdir, filename)

        with h5py.File(outpath, "w") as f:
            # ----------------------------
            # Main averaged data
            # ----------------------------
            f.create_dataset("xi", data=xi_avg)
            f.create_dataset("cov", data=cov_avg)

            # ----------------------------
            # Copy metadata from first file
            # ----------------------------
            for key, val in meta["data"].items():
                dset = f.create_dataset(key, data=val)
                for aname, aval in meta["attrs"][key].items():
                    dset.attrs[aname] = aval

            # ----------------------------
            # Automatically write stats with units
            # ----------------------------
            for key, val in stats.items():
                # if val has units (unyt or astropy.units), save as value + unit string
                if hasattr(val, "units"):
                    dset = f.create_dataset(key, data=val.value)
                    dset.attrs["units"] = str(val.units)
                else:
                    f.create_dataset(key, data=val)

        print(f"Wrote: {outpath}")


    # ---------------------------------------------------------
    # MAIN DRIVER: Process all filenames
    # ---------------------------------------------------------
    def _run_all(self):
        """
        Process all unique filenames found under run_xxx/.

        Produces:
            base_dir/average/<filename>
        """

        print(f"Found {len(self.file_groups)} unique filenames:")
        for fn in self.file_groups:
            print(f"  • {fn} ({len(self.file_groups[fn])} files)")

        print("\nStarting averaging...\n")

        for filename, filepaths in self.file_groups.items():
            print(f"→ {filename}: averaging {len(filepaths)} files")

            xi_avg, cov_avg, meta,stats = self._process_group(
                filename, filepaths
            )

            self._write_output(filename, xi_avg, cov_avg, meta,stats)

        print("\n✔ All filenames processed.")
