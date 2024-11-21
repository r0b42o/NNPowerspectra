"""
Compute and save projected bispectra for galaxy-galaxy (kgg) and galaxy-convergence (kkg) correlations 
for various numbers of steps in the Limber integration
using a halo model and halo occupation distribution (HOD).

This script defines a cosmological model, halo model, and HOD parameters, computes 
bispectra for various angular frequencies (ells) and azimuthal angles (phis) across 
multiple redshift bin configurations, and saves the results to files.

Functions:
- normalize_distribution: Normalizes a redshift distribution to integrate to unity.
- initialize_halo_model: Sets up the halo model, mass function, halo bias, concentration-mass 
  relation, and HOD parameters.
- compute_bispectra: Computes bispectra for cpnvergence-galaxy-galaxy (C_kgg) and convergence -convergence-galaxy
  (C_kkg) correlations for given ells, phis, and redshift bins.
- main: Orchestrates the bispectrum computation and saves the results.

Parameters:
- NZ_POINTS: List of redshift bin counts to evaluate (e.g., [10, 20, 30, ...]).
- OUT_FN_KGG: File path to save the galaxy-galaxy bispectra.
- OUT_FN_KKG: File path to save the galaxy-convergence bispectra.
- COSMO_PARAMS: Dictionary containing cosmological parameters (e.g., Omega_m, h, sigma_8, etc.).
- Z_RANGE: Redshift range for lens and source distributions.
- L_STEPS: Number of ell steps for bispectrum calculation.
- PHI_STEPS: Number of azimuthal angle steps for bispectrum calculations.

Outputs:
- OUT_FN_KGG: A text file containing the computed galaxy-galaxy bispectra.
- OUT_FN_KKG: A text file containing the computed galaxy-convergence bispectra.

Usage:
Run this script in an environment with the required dependencies installed (g3lhalo, pyccl, numpy). 
The results will be saved in the `accuracy_files` directory.

Dependencies:
- g3lhalo: For halo model and bispectrum computations.
- pyccl: For cosmology calculations.
- numpy: For numerical operations.


This will produce two output files: `z_bispec_kgg.dat` and `z_bispec_kkg.dat`.
"""


import g3lhalo
import pyccl as ccl
import numpy as np
from itertools import product

# Constants and Configurations
NZ_POINTS = [10, 20, 30, 50, 100, 200]
OUT_FN_KGG = "./accuracy_files/z_bispec_kgg.dat"
OUT_FN_KKG = "./accuracy_files/z_bispec_kkg.dat"
COSMO_PARAMS = {"Om_c": 0.28, "Om_b": 0.05, "h": 0.67, "sigma_8": 0.8, "n_s": 0.96}
Z_RANGE = np.linspace(0.2, 1.5, 100)
L_STEPS = 5
PHI_STEPS = 5

# Functions
def normalize_distribution(zs, center, width):
    """Generate and normalize a Gaussian redshift distribution."""
    dist = np.exp(-(zs - center) ** 2 / width ** 2)
    return dist / (np.sum(dist) * (zs[1] - zs[0]))

def initialize_halo_model():
    """Initialize the halo model and HOD parameters."""
    hmf = ccl.halos.MassFuncSheth99()
    hbf = ccl.halos.HaloBiasSheth01()
    cmfunc = ccl.halos.ConcentrationDuffy08()
    hod_cen, hod_sat = g3lhalo.HOD_Zheng(
        alpha=1, Mth=1e12, sigma=0.2, Mprime=20 * 1e12, beta=1
    )
    model = g3lhalo.halomodel(
        verbose=True, cosmo=COSMO_PARAMS, hmfunc=hmf, hbfunc=hbf, cmfunc=cmfunc
    )
    model.set_hods(hod_cen, hod_sat, A=0, epsilon=0, flens1=1, flens2=1)
    return model

def compute_bispectra(model, nzpoints, ells1, ells2, phis, zs, n_lenses, n_sources):
    """Compute bispectra for galaxy-galaxy and galaxy-convergence correlations."""
    bs_kgg = np.zeros((len(nzpoints), len(ells1) * len(ells2) * len(phis)))
    bs_kkg = np.zeros((len(nzpoints), len(ells1) * len(ells2) * len(phis)))

    for iz, nz in enumerate(nzpoints):
        print(f"Processing {nz} redshift bins...")
        projection = g3lhalo.projectedSpectra(model, nzbins=nz)
        projection.set_nz_lenses(zs, n_lenses)
        projection.set_nz_sources(zs, n_sources)

        for count, (ell1, ell2, phi) in enumerate(product(ells1, ells2, phis)):
            ell3 = np.sqrt(ell1**2 + ell2**2 + 2 * np.cos(phi) * ell1 * ell2)
            _, _, _, bs_kgg[iz, count] = projection.C_kgg(ell1, ell2, ell3, 1, 1)
            _, _, _, bs_kkg[iz, count] = projection.C_kkg(ell1, ell2, ell3, 1)
    
    return bs_kgg, bs_kkg

def main():
    """Main function to orchestrate the bispectrum computation."""
    zs = Z_RANGE
    n_sources = normalize_distribution(zs, center=1.0, width=0.02)
    n_lenses = normalize_distribution(zs, center=0.5, width=0.02)

    model = initialize_halo_model()
    ells1 = np.geomspace(1, 20000, L_STEPS)
    ells2 = np.geomspace(1, 20000, L_STEPS)
    phis = np.linspace(0, 2 * np.pi, PHI_STEPS)

    bs_kgg, bs_kkg = compute_bispectra(
        model, NZ_POINTS, ells1, ells2, phis, zs, n_lenses, n_sources
    )

    np.savetxt(OUT_FN_KGG, bs_kgg)
    np.savetxt(OUT_FN_KKG, bs_kkg)

if __name__ == "__main__":
    main()
