"""
Script to compute bispectra for galaxy-galaxy (kgg) and galaxy-convergence (kkg) correlations
and derive aperture statistics using a halo model.

This script performs the following steps:
1. Initializes the cosmology and halo model with Sheth-Tormen mass function and bias, 
   Duffy concentration-mass relation, and Zheng et al. HOD parameters.
2. Defines source and lens redshift distributions and sets them in the projection model.
3. Computes bispectra (kgg and kkg) for varying resolutions (number of ell bins and angles).
4. Calculates aperture statistics for each resolution and saves results to output files.

Functions:
- normalize_distribution: Normalizes a redshift distribution to integrate to unity.
- compute_bispectrum: Calculates bispectra for kgg or kkg given ell and phi values.
- compute_aperture_statistics: Derives aperture statistics from a bispectrum dictionary.
- save_results: Saves the computed results (theta, aperture statistics) to a text file.

Parameters:
- COSMO_PARAMS: Dictionary of cosmological parameters (e.g., Omega_m, h, sigma_8, etc.).
- HALO_MODEL_PARAMS: Parameters for the HOD (e.g., alpha, Mth, sigma, Mprime, beta).
- OUTPUT_DIR: Directory to save the results.
- ELL_MAX: Maximum angular frequency for the bispectrum calculations.
- NUM_THETAS: Number of theta values for aperture statistics.
- THETA_RANGE: Min and max theta values for aperture statistics in arcminutes.
- ZS_RANGE: Redshift range for lens and source distributions.

Outputs:
- For each resolution `N`:
  - MNN_N_<N>.dat: Aperture statistics for kgg bispectra.
  - MMN_N_<N>.dat: Aperture statistics for kkg bispectra.

Dependencies:
- g3lhalo: For halo model, HOD, bispectra, and aperture statistics calculations.
- pyccl: For cosmological computations.
- numpy: For numerical operations.

Usage:
1. Install required dependencies (g3lhalo, pyccl, numpy).
2. Run the script in a Python environment.
3. Results will be saved in the specified `OUTPUT_DIR`.

Example:
$ python compute_bispectra_and_apstats.py

This will generate output files for each resolution in the format:
- `MNN_N_<N>.dat` (kgg aperture statistics)
- `MMN_N_<N>.dat` (kkg aperture statistics)
"""


import g3lhalo
import pyccl as ccl
import numpy as np

# Constants and Configurations
COSMO_PARAMS = {"Om_c": 0.28, "Om_b": 0.05, "h": 0.67, "sigma_8": 0.8, "n_s": 0.96}
HALO_MODEL_PARAMS = {"alpha": 1, "sigma": 0.2, "Mth": 1e12, "Mprime": 20e12, "beta": 1}
OUTPUT_DIR = "./accuracy_files/"
ELL_MAX = 20000
NUM_THETAS = 10
THETA_RANGE = (2, 200)
ZS_RANGE = np.linspace(0.2, 1.5, 100)

# Functions
def normalize_distribution(zs, center, width):
    """Normalize Gaussian redshift distribution."""
    dist = np.exp(-((zs - center) ** 2) / width ** 2)
    return dist / (np.sum(dist) * (zs[1] - zs[0]))

def compute_bispectrum(projection, ells1, ells2, phis, method):
    """Compute bispectrum for given ell and phi values."""
    bispectrum = np.zeros((len(ells1), len(ells2), len(phis)))

    for i, ell1 in enumerate(ells1):
        for j, ell2 in enumerate(ells2):
            for k, phi in enumerate(phis):
                ell3 = np.sqrt(ell1**2 + ell2**2 + 2 * np.cos(phi) * ell1 * ell2)
                if method == "kgg":
                    _, _, _, B = projection.C_kgg(ell1, ell2, ell3, 1, 1)
                elif method == "kkg":
                    _, _, _, B = projection.C_kkg(ell1, ell2, ell3, 1)
                bispectrum[i, j, k] = B

    return bispectrum

def compute_aperture_statistics(bispectrum_dict, thetas):
    """Compute aperture statistics from bispectrum."""
    aperture_calculator = g3lhalo.apertureStatistics(bispec=bispectrum_dict)
    aperture_stats = []
    for theta in thetas:
        theta_rad = np.deg2rad(theta / 60)
        aperture_stats.append(
            aperture_calculator.third_order(theta_rad, theta_rad, theta_rad, num_points=200)
        )
    return aperture_stats

def save_results(filename, thetas, statistics):
    """Save computed statistics to file."""
    np.savetxt(filename, np.column_stack([thetas, statistics]))

def main():
    # Initialize cosmology and halo model
    cosmo = COSMO_PARAMS
    hmf = ccl.halos.MassFuncSheth99()
    hbf = ccl.halos.HaloBiasSheth01()
    cmfunc = ccl.halos.ConcentrationDuffy08()

    hod_cen, hod_sat = g3lhalo.HOD_Zheng(**HALO_MODEL_PARAMS)
    model = g3lhalo.halomodel(
        verbose=True, cosmo=cosmo, hmfunc=hmf, hbfunc=hbf, cmfunc=cmfunc
    )
    model.set_hods(hod_cen, hod_sat, A=0, epsilon=0, flens1=1, flens2=1)

    # Define redshift distributions
    zs = ZS_RANGE
    n_sources = normalize_distribution(zs, center=1.0, width=0.02)
    n_lenses = normalize_distribution(zs, center=0.5, width=0.02)

    # Projection setup
    projection = g3lhalo.projectedSpectra(model, nzbins=50)
    projection.set_nz_lenses(zs, n_lenses)
    projection.set_nz_sources(zs, n_sources)

    # Loop over different resolutions
    for N in [5, 10, 20, 30, 50]:
        print(f"Processing resolution N={N}")

        ells1 = np.geomspace(1, ELL_MAX, N)
        ells2 = np.geomspace(1, ELL_MAX, N)
        phis = np.linspace(0, 2 * np.pi, N)

        # Compute kgg bispectrum and aperture statistics
        Bs_kgg = compute_bispectrum(projection, ells1, ells2, phis, method="kgg")
        kgg_dict = {"ell1": ells1, "ell2": ells2, "phi": phis, "bs": Bs_kgg}

        thetas = np.geomspace(*THETA_RANGE, NUM_THETAS)
        MapNapNap = compute_aperture_statistics(kgg_dict, thetas)
        save_results(f"{OUTPUT_DIR}/MNN_N_{N}.dat", thetas, MapNapNap)

        # Compute kkg bispectrum and aperture statistics
        Bs_kkg = compute_bispectrum(projection, ells1, ells2, phis, method="kkg")
        kkg_dict = {"ell1": ells1, "ell2": ells2, "phi": phis, "bs": Bs_kkg}

        MapMapNap = compute_aperture_statistics(kkg_dict, thetas)
        save_results(f"{OUTPUT_DIR}/MMN_N_{N}.dat", thetas, MapMapNap)

if __name__ == "__main__":
    main()
