# BAO DESI Analysis Pipeline

This repository contains the analysis pipeline developed for my Baryon Acoustic Oscillation (BAO) study using DESI-like mock galaxy catalogues. The project investigates how cosmological parameters — particularly dark energy models — affect the recovered BAO scale.

The pipeline extracts the BAO dilation parameter α, compares it to the fiducial cosmology, and studies potential systematic biases across redshift.

---

## Project Overview

Baryon Acoustic Oscillations provide a robust standard ruler for measuring the expansion history of the Universe. In this project, I:

- Construct lightcone catalogues from simulation snapshots  
- Measure the galaxy two-point correlation function  
- Fit a BAO template to extract the dilation parameter α  
- Compare recovered values to the fiducial cosmology  
- Investigate redshift evolution and systematic trends  

The analysis is motivated by current and upcoming surveys such as DESI, which aim to constrain dark energy through precise BAO measurements.

---

## Cosmological Models Studied

The pipeline supports testing different dark energy scenarios:

- ΛCDM (cosmological constant, w = -1)
- Quintessence (w > -1)
- Phantom dark energy (w < -1)
- CPL parametrization:

  w(a) = w₀ + wₐ (1 − a)

Note: The CPL parametrization is treated purely as a first-order Taylor expansion in scale factor and is not assumed to arise from a specific physical model.

---

## Pipeline Structure

The workflow consists of:

### 1. Catalogue Preparation
- Load simulation outputs  
- Apply redshift cuts  
- Construct lightcone geometry  

### 2. Distance Calculations
- Compute comoving distances  
- Implement custom cosmology module  
- Control parameter variations (Ω_m, Ω_de, curvature)

### 3. Clustering Measurement
- Two-point correlation function using KD-tree pair counting  
- Landy–Szalay estimator  

### 4. BAO Fitting
- Template fitting procedure  
- Extraction of dilation parameter α  
- Error estimation via covariance matrix  

### 5. Diagnostics
- Redshift evolution of α  
- Systematic bias checks  
- Comparison to input cosmology  

---

## Key Result

Across redshift bins, recovered values of α are generally close to unity, but a mild preference for α < 1 appears at higher redshift. This may indicate:

- Residual fitting systematics  
- Cosmology–template mismatch  
- Lightcone construction effects  

Further investigation is ongoing.

---

## Dependencies

- Python 3.x  
- numpy>=1.27
- treecorr
- unyt
- swiftsimio
- psutil
- numba
- matplotlib
- scipy
- PyYAML
- Corrfunc (for faster pair counting)
  


---

## How to Run (COSMA Cluster)

The pipeline is designed to run on the COSMA HPC cluster using SLURM job submission.  
All simulation inputs, cosmological parameters, and output directories are defined in the YAML configuration file.

Example configuration file:

    configurations/w0wa_real.yaml

This file controls:

- Input simulation path  
- Cosmological parameters (Ω_m, Ω_de, w0, wa, etc.)  
- Redshift ranges  
- Output directory for plots and data products  
- Analysis settings  

Always verify the configuration file before submitting jobs.

---

## Step 1 — Run BAO Realisations (MPI + SLURM Array)

The main computation (pair counting + BAO fitting per realization) is executed as a SLURM job array.

Submit:

    sbatch run_w0wa.sh

This script:

- Launches a SLURM job array (`--array=0-9`)
- Exports `REALIZATION_ID` from `SLURM_ARRAY_TASK_ID`
- Runs 101 MPI tasks on a single node
- Executes:

    mpirun -np $SLURM_NTASKS python -u main.py --config configurations/w0wa_real.yaml

Each array index corresponds to one independent realization.

Log files are written to:

    logs_w0wa_<jobID>/run_<arrayID>.out
    logs_w0wa_<jobID>/run_<arrayID>.err

All numerical outputs and plots are written to the directory specified in the config file (`output_dir`).

---

## Step 2 — Average the Realisations

After all realizations have successfully completed, run the averaging stage:

    sbatch run_averaging.sh

This executes:

    python average_run.py --config configurations/w0wa_real.yaml

The averaging script:

- Reads all realization outputs from the configured output directory  
- Computes the mean correlation function  
- Produces final averaged BAO measurements  
- Generates the final plots  

Logs are written to:

    logs_averaging_real_<jobID>/

---

## Environment Setup (COSMA)

The SLURM scripts assume:

- Virtual environment:

      ~/my_env/

- Required modules:

      module purge
      module load gnu_comp/13.1.0
      module load openmpi/4.1.4

- Working directory:

      /cosma/home/<project>/<user>/BAO

---

## Full Workflow Summary

1. Configure simulation input, cosmology, and output directory in:
   
       configurations/w0wa_real.yaml

2. Submit MPI job array:
   
       sbatch run_w0wa.sh

3. Wait for all realizations to finish.

4. Submit averaging job:
   
       sbatch run_averaging.sh

5. Retrieve final plots and data products from the output directory specified in the configuration file.

---

This two-stage workflow separates computationally intensive clustering measurements from the final statistical aggregation, allowing scalable production of BAO constraints across multiple cosmological models.


---

## Scientific Context

BAO measurements provide one of the cleanest probes of dark energy. This project aims to:

- Test robustness of BAO recovery  
- Quantify cosmology-dependent biases  
- Explore dynamical dark energy effects  



---

## Author

Stan Kortmann 
MPhys Physics  
Leiden University, Lorentz Instituut 
