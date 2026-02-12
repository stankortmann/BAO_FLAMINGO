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
  

Optional:
- Corrfunc (for faster pair counting)  
- emcee (for MCMC fitting)

---

## How to Run

1. Clone the repository:
   git clone https://github.com/stankortmann/BAO_FLAMINGO.git  
   cd BAO   

2. Run the main pipeline with a configuration file:
   python run_pipeline.py  --config configurations/configuration.yaml

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
