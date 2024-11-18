import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici
import scipy.integrate as integrate

epsrel=1e-4

N=int(256)

class halomodel:

    def __init__(
        self, verbose=False, cosmo=None, hmfunc=None, hbfunc=None, cmfunc=None
    ):
        """Halomodel initialization. Automatically assigns cosmology, halo mass function, halo bias and concetration mass relation if these are specified.

        Args:
            verbose (bool, optional): Control verbosity. Defaults to False.
            cosmo (dict, optional): Cosmological parameters as dictionary. Needs to include 'Om_b', 'Om_c', 'h', 'sigma_8', 'n_s'. Other parameters are ignored. Defaults to None.
            hmfunc (function, optional): Halo mass function [dn/dlog10(M/Msun)]. Needs to be function of halo mass and scale factor. Defaults to None.
            hbfunc (function, optional): Halo bias function. Needs to be function of halo mass and scale factor. Defaults to None.
            cmfunc (function, optional): Concentration mass relation. Needs to be function of halo mass and scale factor. Defaults to None.
        """

        self.verbose = verbose

        if type(cosmo) is dict:
            self.set_cosmo(cosmo)

        if callable(hmfunc):
            self.set_hmf(hmfunc)

        if callable(hbfunc):
            self.set_halobias(hbfunc)

        if callable(cmfunc):
            self.set_concentration_mass_relation(cmfunc)

    def set_cosmo(self, cosmo):
        """Sets cosmology

        Args:
            cosmo (dict): Cosmological parameters as dictionary. Needs to include 'Om_b', 'Om_c', 'h', 'sigma_8', 'n_s'. Other parameters are ignored.
        """

        if self.verbose:
            print("Setting cosmology")
            print(f"Om_c: {cosmo['Om_c']}")
            print(f"Om_b: {cosmo['Om_b']}")
            print(f"h: {cosmo['h']}")
            print(f"sigma_8: {cosmo['sigma_8']}")
            print(f"n_s: {cosmo['n_s']}")

        self.cosmo = ccl.Cosmology(
            Omega_c=cosmo["Om_c"],
            Omega_b=cosmo["Om_b"],
            h=cosmo["h"],
            sigma8=cosmo["sigma_8"],
            n_s=cosmo["n_s"],
        )

        if self.verbose:
            print("Also setting linear matter power spectrum")

        self.pk_lin = lambda k, z: ccl.linear_matter_power(
            self.cosmo, k, 1.0 / (1.0 + z)
        )

    def set_hmf(self, hmfunc):
        """Sets halo mass function. Note that the internal model.dndm is in units of 1/Msun/Mpc³ and a function of mass and redshift not scale factor!

        Args:
            hmfunc (function): Halo mass function [dn/dlog10(M/Msun)]. Needs to be function of halo mass and scale factor.
        """

        if self.verbose:
            print("Setting halo mass function")
            print(hmfunc)

        self.dndm = lambda m, z: hmfunc(self.cosmo, m, 1.0 / (1.0 + z)) / (
            m * np.log(10)
        )

    def set_halobias(self, hbfunc):
        """Sets halo bias function. The internal model.bh is a function of halo mass and redshift

        Args:
            hbfunc (function): Halo bias function. Needs to be function of halo mass and scale factor
        """
        if self.verbose:
            print("Setting halo bias function")
            print(hbfunc)
        self.bh = lambda m, z: hbfunc(self.cosmo, m, 1.0 / (1.0 + z))

    def set_concentration_mass_relation(self, cmfunc):
        """Sets concentration-mass relation. The internal model.cm is a function of halo mass and redshift

        Args:
            cmfunc (function): Concentration-mass relation. Needs to be function of halo mass and scale factor
        """
        if self.verbose:
            print("Setting concentration mass relation")
            print(cmfunc)
        self.cm = lambda m, z: cmfunc(self.cosmo, m, 1.0 / (1.0 + z))

    def set_hod1(self, hodfunc_cen, hodfunc_sat):
        """Sets HOD for first lens population

        Args:
            hodfunc_cen (function): Central galaxy number as function of halo mass
            hodfunc_sat (function): Satellite galaxy number as function of halo mass
        """
        self.hod_cen1 = hodfunc_cen
        self.hod_sat1 = hodfunc_sat

    def set_hod2(self, hodfunc_cen, hodfunc_sat):
        """Sets HOD for second lens population

        Args:
            hodfunc_cen (function): Central galaxy number as function of halo mass
            hodfunc_sat (function): Satellite galaxy number as function of halo mass
        """
        self.hod_cen2 = hodfunc_cen
        self.hod_sat2 = hodfunc_sat

    def set_hod_corr(self, A=0, epsilon=0):
        """Sets correlation of the HODs of the two populations

        Args:
            A (float): Amplitude of correlation
            epsilon (float): Power law Index of correlation
        """
        self.hod_satsat = lambda m: self.hod_sat1(m) * self.hod_sat2(m) + A * np.power(
            m, epsilon
        ) * np.sqrt(self.hod_sat1(m) * self.hod_sat2(m))

    def set_hods(
        self,
        hodfunc_cen1,
        hodfunc_sat1,
        hodfunc_cen2=None,
        hodfunc_sat2=None,
        A=0,
        epsilon=0,
        flens1=1,
        flens2=1,
    ):
        """ Sets HODs

        Args:
            hodfunc_cen1 (function): Central galaxy number as function of halo mass for first population
            hodfunc_sat1 (function): Satellite galaxy number as function of halo mass for first population
            hodfunc_cen2 (function, optional): Central galaxy number as function of halo mass for second population. Defaults to None, then population 2 = population 1.
            hodfunc_sat2 (function, optional): Satellite galaxy number as function of halo mass for first population. Defaults to None, then population 2 = population 1.
            A (float, optional): Amplitude of correlation. Defaults to 0.
            epsilon (float, optional): Power law index of correlation. Defaults to 0.
            flens1 (float, optional): Scaling of halo profile correlation for first lens population. Defaults to 1, then same as dark matter concentration.
            flens2 (float, optional): Scaling of halo profile correlation for second lens population. Defaults to 1, then same as dark matter concentration.
        """
        self.set_hod1(hodfunc_cen1, hodfunc_sat1)
        if hodfunc_cen2 != None:
            self.set_hod2(hodfunc_cen2, hodfunc_sat2)
        else:
            self.set_hod2(hodfunc_cen1, hodfunc_sat1)

        self.set_hod_corr(A, epsilon)

        self.flens1 = flens1
        self.flens2 = flens2

    def u_NFW(self, k, m, z, f=1.0):
        """Fourier-transformed, normalized NFW halo profile

        Args:
            k (float): wavenumber [1/Mpc]
            m (float): halo mass [Msun]
            z (float): redshift
            f (float, optional): Scaling of halo profile concentration. Defaults to 1.0.
        """
        a = 1 / (1 + z)
        c = f * self.cm(m, z)

        r200 = np.power(0.239 * m / 200 / ccl.rho_x(self.cosmo, a, "matter"), 1.0 / 3.0)

        arg1 = k * r200 / c
        arg2 = arg1 * (1 + c)

        si1, ci1 = sici(arg1)
        si2, ci2 = sici(arg2)

        term1 = np.sin(arg1) * (si2 - si1)
        term2 = np.cos(arg1) * (ci2 - ci1)
        term3 = -np.sin(arg1 * c) / arg2

        F = np.log(1.0 + c) - c / (1.0 + c)

        return (term1 + term2 + term3) / F

    def G_ab(self, k1, k2, m, z, type1=1, type2=2):
        """Helping function for lens-lens correlations. See Eq. 56 of Linke, Simon, Schneider+ (2023)

        Args:
            k1 (float): Wavenumber [1/Mpc]
            k2 (float): Wavenumber [1/Mpc]
            m (float): Halo mass [Msun]
            z (float): Redshift
            type1 (int, optional): Which is the first lens population. Defaults to 1.
            type2 (int, optional): Which is the second lens population. Defaults to 2.
        """

        if type1 == 1:
            Nc1 = self.hod_cen1(m)
            Ns1 = self.hod_sat1(m)
            u1 = self.u_NFW(k1, m, z, self.flens1)

        else:
            Nc1 = self.hod_cen2(m)
            Ns1 = self.hod_cen2(m)
            u1 = self.u_NFW(k1, m, z, self.flens2)

        if type2 == 1:
            Nc2 = self.hod_cen1(m)
            Ns2 = self.hod_sat1(m)
            u2 = self.u_NFW(k2, m, z, self.flens1)

        else:
            Nc2 = self.hod_cen2(m)
            Ns2 = self.hod_cen2(m)
            u2 = self.u_NFW(k2, m, z, self.flens2)

        if type1 == type2:
            Nss = (Ns1 - 1) * Ns1
        else:
            Nss = self.hod_satsat(m)

        return Nc1 * Ns2 * u2 + Nc2 * Ns1 * u1 + Nss * u1 * u2

    def G_a(self, k, m, z, type=1):
        """Helping function for lens correlation. See Eq. 55 of Linke, Simon, Schneider+ (2023)

        Args:
            k (float): wavenumber [1/Mpc]
            m (float): Halo mass [Msun]
            z (float): redshift
            type (int, optional): Which lens population to use. Defaults to 1.
        """
        if type == 1:
            Nc = self.hod_cen1(m)
            Ns = self.hod_sat1(m)
            u = self.u_NFW(k, m, z, self.flens1)

        else:
            Nc = self.hod_cen2(m)
            Ns = self.hod_cen2(m)
            u = self.u_NFW(k, m, z, self.flens2)

        return Nc + Ns * u

    def get_ave_numberdensity(self, z, type=1, mmin=1e10, mmax=1e17):
        """Calculates the average galaxy number density from the HOD

        Args:
            z (float): redshift
            type (int, optional): Which lens population to use. Defaults to 1.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.

        Returns:
            float: average number density [1/Mpc³]
        """
        if type == 1:
            Nc = self.hod_cen1
            Ns = self.hod_sat1

        else:
            Nc = self.hod_cen2
            Ns = self.hod_cen2

        kernel = lambda m: (Nc(m) + Ns(m)) * self.dndm(m, z)

        return np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

    def lens_lens_ps_1h(self, ks, z, type1=1, type2=2, mmin=1e10, mmax=1e17):
        """ 1-halo term of Lens-Lens Powerspectrum (not normalized by number density!)

        Args:
            ks (float): wavenumber [1/Mpc]
            z (float): redshift
            type1 (int, optional): Which first lens population. Defaults to 1.
            type2 (int, optional): Which second lens population. Defaults to 2.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """

        ks = np.array(ks)
        integral = []

        for k in ks:
            kernel = lambda m: self.G_ab(k, k, m, z, type1, type2) * self.dndm(m, z)

            integral.append(np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N)))

        integral = np.array(integral)
        return integral

    def lens_lens_ps_2h(self, ks, z, type1=1, type2=2, mmin=1e10, mmax=1e17):
        """ 2-halo term of Lens-Lens Powerspectrum (not normalized by number density!)

        Args:
            ks (float): wavenumber [1/Mpc]
            z (float): redshift
            type1 (int, optional): Which first lens population. Defaults to 1.
            type2 (int, optional): Which second lens population. Defaults to 2.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """

        ks = np.array(ks)

        integral1 = []

        for k in ks:
            kernel = (
                lambda m: self.G_a(k, m, z, type1) * self.dndm(m, z) * self.bh(m, z)
            )
            integral1.append(np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N)))

        integral1 = np.array(integral1)

        if type1 == type2:
            integral2 = integral1
        else:
            integral2 = []

            for k in ks:
                kernel = (
                    lambda m: self.G_a(k, m, z, type2) * self.dndm(m, z) * self.bh(m, z)
                )
                integral2.append(np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N)))
            integral2 = np.array(integral2)

        return integral1 * integral2 * self.pk_lin(ks, z)

    def lens_lens_ps(self, ks, z, type1=1, type2=2, mmin=1e10, mmax=1e17):
        """ Lens-Lens Powerspectrum
        Args:
            ks (float): wavenumber [1/Mpc]
            z (float): redshift
            type1 (int, optional): Which first lens population. Defaults to 1.
            type2 (int, optional): Which second lens population. Defaults to 2.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.

        Return:
            float: normalized 1-halo term
            float: normalized 2-halo term
            float: total power spectrum
        """

        n1 = self.get_ave_numberdensity(z, type1)
        if type1 == type2:
            n2 = n1
        else:
            n2 = self.get_ave_numberdensity(z, type2)

        One_halo = self.lens_lens_ps_1h(ks, z, type1, type2, mmin, mmax) / n1 / n2
        Two_halo = self.lens_lens_ps_2h(ks, z, type1, type2, mmin, mmax) / n1 / n2

        return One_halo, Two_halo, One_halo + Two_halo

    def source_lens_ps_1h(self, ks, z, type=1, mmin=1e10, mmax=1e17):
        """ 1-halo term of Source-Lens Powerspectrum (not normalized by number density!)

        Args:
            ks (float): wavenumber [1/Mpc]
            z (float): redshift
            type (int, optional): Which lens population. Defaults to 1.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """

        ks = np.array(ks)

        integral = []
        for k in ks:
            kernel = (
                lambda m: self.G_a(k, m, z, type)
                * self.dndm(m, z)
                * m
                * self.u_NFW(k, m, z, 1.0)
            )
            integral.append(np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N)))

        integral = np.array(integral)
        return integral

    def source_lens_ps_2h(self, ks, z, type=1, mmin=1e10, mmax=1e17):
        """ 2-halo term of Source-Lens Powerspectrum (not normalized by number density!). 
        Includes halo mass integral correction from Appendix B of Mead+ (2020)

        Args:
            ks (float): wavenumber [1/Mpc]
            z (float): redshift
            type (int, optional): Which lens population. Defaults to 1.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """

        ks = np.array(ks)

        integral1 = []
        for k in ks:
            kernel = lambda m: self.G_a(k, m, z, type) * self.dndm(m, z) * self.bh(m, z)
            integral1.append(np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N)))
        integral1 = np.array(integral1)

        integral2 = []
        for k in ks:
            kernel = lambda m: self.u_NFW(k, m, z, 1.0) * self.dndm(m, z) * m
            integral2.append(np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N)))
        integral2 = np.array(integral2)

        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")
        corr2 = (rho_bar - self.A_2h_correction(z, mmin, mmax)) * self.u_NFW(
            ks, mmin, z, f=1.0
        )
        integral2 += corr2

        return integral1 * integral2 * self.pk_lin(ks, z)

    def source_lens_ps(self, ks, z, type=1, mmin=1e10, mmax=1e17):
        """ Source-Lens Powerspectrum
        Args:
            ks (float): wavenumber [1/Mpc]
            z (float): redshift
            type (int, optional): Which lens population. Defaults to 1.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.

        Return:
            float: normalized 1-halo term
            float: normalized 2-halo term
            float: total power spectrum
        """
        n = self.get_ave_numberdensity(z, type)
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = self.source_lens_ps_1h(ks, z, type, mmin, mmax) / n / rho_bar
        Two_halo = self.source_lens_ps_2h(ks, z, type, mmin, mmax) / n / rho_bar

        return One_halo, Two_halo, One_halo + Two_halo

    def source_source_ps_1h(self, ks, z, mmin=1e10, mmax=1e17):
        """ 1-halo term of Source-Source Powerspectrum (not normalized by matter density!)

        Args:
            ks (float): wavenumber [1/Mpc]
            z (float): redshift
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """

        ks = np.array(ks)

        integral = []
        for k in ks:
            kernel = lambda m: self.dndm(m, z) * m * m * (self.u_NFW(k, m, z)) ** 2
            integral.append(np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N)))

        integral = np.array(integral)
        return integral

    def source_source_ps_2h(self, ks, z, mmin=1e10, mmax=1e17):
        """ 2-halo term of Source-Source Powerspectrum (not normalized by matter density!)
        Includes halo mass integral correction from Appendix B of Mead+ (2020)

        Args:
            ks (float): wavenumber [1/Mpc]
            z (float): redshift
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """
        ks = np.array(ks)

        integral = []

        for k in ks:
            kernel = lambda m: self.u_NFW(k, m, z) * self.dndm(m, z) * m * self.bh(m, z)
            integral.append(np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N)))

        integral = np.array(integral)
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        corr = (rho_bar - self.A_2h_correction(z, mmin, mmax)) * self.u_NFW(ks, mmin, z)
        integral += corr

        return integral**2 * self.pk_lin(ks, z)

    def A_2h_correction(self, z, mmin=1e10, mmax=1e17):
        """Amplitude of mass integral correction from Appendix B of Mead+ (2020)

        Args:
            z (float): redshift
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17
        """
        kernel_A = lambda m: self.dndm(m, z) * self.bh(m, z) * m
        A = integrate.quad(kernel_A, mmin, mmax, epsrel=epsrel)[0]
        return A

    def source_source_ps(self, ks, z):
        """ Source-Source Powerspectrum
        Args:
            ks (float): wavenumber [1/Mpc]
            z (float): redshift
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.

        Return:
            float: normalized 1-halo term
            float: normalized 2-halo term
            float: total power spectrum
        """
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")
        One_halo = self.source_source_ps_1h(ks, z) / rho_bar**2
        Two_halo = self.source_source_ps_2h(ks, z) / rho_bar**2

        return One_halo, Two_halo, One_halo + Two_halo

    def source_lens_lens_bs_1h(
        self, k1, k2, k3, z, type1=1, type2=2, mmin=1e10, mmax=1e17
    ):
        """1-halo term of Source-Lens-Lens Bispectrum (not normalized by number density!)

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            type1 (int, optional): Which first lens population. Defaults to 1.
            type2 (int, optional): Which second lens population. Defaults to 2.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """
        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * self.u_NFW(k1, m, z, 1.0)
            * self.G_ab(k2, k3, m, z, type1, type2)
        )
        integral = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        return integral

    def source_lens_lens_bs_2h(
        self, k1, k2, k3, z, type1=1, type2=2, mmin=1e10, mmax=1e17
    ):
        """2-halo term of Source-Lens-Lens Bispectrum (not normalized by number density!)

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            type1 (int, optional): Which first lens population. Defaults to 1.
            type2 (int, optional): Which second lens population. Defaults to 2.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """
        kernel = (
            lambda m: self.dndm(m, z) * self.u_NFW(k1, m, z, 1.0) * m * self.bh(m, z)
        )
        summand1 = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_ab(k2, k3, m, z, type1, type2)
            * self.bh(m, z)
        )
        summand1 *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        summand1 *= self.pk_lin(k3, z)

        kernel = lambda m: self.dndm(m, z) * self.G_a(k2, m, z, type1) * self.bh(m, z)
        summand2 = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_a(k3, m, z, type2)
            * m
            * self.u_NFW(k1, m, z)
            * self.bh(m, z)
        )
        summand2 *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        summand2 *= self.pk_lin(k1, z)

        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, z, type2) * self.bh(m, z)
        summand3 = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_a(k2, m, z, type1)
            * m
            * self.u_NFW(k1, m, z)
            * self.bh(m, z)
        )
        summand3 *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        summand3 *= self.pk_lin(k2, z)

        return summand1 + summand2 + summand3

    def source_lens_lens_bs_3h(
        self, k1, k2, k3, z, type1=1, type2=2, mmin=1e10, mmax=1e17
    ):
        """3-halo term of Source-Lens-Lens Bispectrum (not normalized by number density!)

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            type1 (int, optional): Which first lens population. Defaults to 1.
            type2 (int, optional): Which second lens population. Defaults to 2.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """
        kernel = lambda m: self.dndm(m, z) * self.G_a(k2, m, z, type1) * self.bh(m, z)
        integral = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, z, type2) * self.bh(m, z)
        integral *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z)
        integral *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        return integral * self.bk_lin(k1, k2, k3, z)

    def source_lens_lens_bs(
        self, k1, k2, k3, z, type1=1, type2=2, mmin=1e10, mmax=1e17
    ):
        """Source-Lens-Lens Bispectrum 

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            type1 (int, optional): Which first lens population. Defaults to 1.
            type2 (int, optional): Which second lens population. Defaults to 2.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.


        Return:
            float: normalized 1-halo term
            float: normalized 2-halo term
            float: normalized 3-halo term
            float: total bispectrum
        """
        n1 = self.get_ave_numberdensity(z, type1)
        if type1 == type2:
            n2 = n1
        else:
            n2 = self.get_ave_numberdensity(z, type2)

        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = (
            self.source_lens_lens_bs_1h(k1, k2, k3, z, type1, type2, mmin, mmax)
            / n1
            / n2
            / rho_bar
        )

        Two_halo = (
            self.source_lens_lens_bs_2h(k1, k2, k3, z, type1, type2, mmin, mmax)
            / n1
            / n2
            / rho_bar
        )

        Three_halo = (
            self.source_lens_lens_bs_3h(k1, k2, k3, z, type1, type2, mmin, mmax)
            / n1
            / n2
            / rho_bar
        )

        return One_halo, Two_halo, Three_halo, One_halo + Two_halo + Three_halo

    def source_source_lens_bs_1h(self, k1, k2, k3, z, type=1, mmin=1e10, mmax=1e17):
        """1-halo term of Source-Source-Lens Bispectrum (not normalized by number density!)

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            type (int, optional): Which lens population. Defaults to 1.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """

        kernel = (
            lambda m: self.dndm(m, z)
            * self.G_a(k3, m, z, type)
            * m
            * m
            * self.u_NFW(k1, m, z)
            * self.u_NFW(k2, m, z)
        )
        integral = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        return integral

    def source_source_lens_bs_2h(self, k1, k2, k3, z, type=1, mmin=1e10, mmax=1e17):
        """2-halo term of Source-Source-Lens Bispectrum (not normalized by number density!)

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            type (int, optional): Which lens population. Defaults to 1.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """

        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, type) * self.bh(m, z)
        summand1 = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * m
            * self.u_NFW(k1, m, z)
            * self.u_NFW(k2, m, z)
            * self.bh(m, z)
        )
        summand1 *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        summand1 *= self.pk_lin(k3, z)

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z) * self.bh(m, z)
        summand2 = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * self.u_NFW(k2, m, z)
            * self.G_a(k3, m, z, type)
            * self.bh(m, z)
        )
        summand2 *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        summand2 *= self.pk_lin(k1, z)

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k2, m, z) * self.bh(m, z)
        summand3 = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = (
            lambda m: self.dndm(m, z)
            * m
            * self.u_NFW(k1, m, z)
            * self.G_a(k3, m, z, type)
            * self.bh(m, z)
        )

        summand3 *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        summand3 *= self.pk_lin(k2, z)

        return summand1 + summand2 + summand3

    #@njit(parallel=True)
    def source_source_lens_bs_3h(self, k1, k2, k3, z, type=1, mmin=1e10, mmax=1e17):
        """3-halo term of Source-Source-Lens Bispectrum (not normalized by number density!)

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            type (int, optional): Which lens population. Defaults to 1.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """
        kernel = lambda m: self.dndm(m, z) * self.G_a(k3, m, z, type) * self.bh(m, z)
        integral = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z) * self.bh(m, z)
        integral *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k2, m, z) * self.bh(m, z)
        integral *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        return integral * self.bk_lin(k1, k2, k3, z)

    def source_source_lens_bs(self, k1, k2, k3, z, type=1, mmin=1e10, mmax=1e17):
        """Source-Source-Lens Bispectrum 

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            type (int, optional): Which lens population. Defaults to 1.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.


        Return:
            float: normalized 1-halo term
            float: normalized 2-halo term
            float: normalized 3-halo term
            float: total bispectrum
        """
        n = self.get_ave_numberdensity(z, type)
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = (
            self.source_source_lens_bs_1h(k1, k2, k3, z, type, mmin, mmax)
            / n
            / rho_bar
            / rho_bar
        )

        Two_halo = (
            self.source_source_lens_bs_2h(k1, k2, k3, z, type, mmin, mmax)
            / n
            / rho_bar
            / rho_bar
        )

        Three_halo = (
            self.source_source_lens_bs_3h(k1, k2, k3, z, type, mmin, mmax)
            / n
            / rho_bar
            / rho_bar
        )

        return One_halo, Two_halo, Three_halo, One_halo + Two_halo + Three_halo

    def source_source_source_bs_1h(self, k1, k2, k3, z, mmin=1e10, mmax=1e17):
        """1-halo term of Source-Source-Source Bispectrum (not normalized by matter density!)

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """

        kernel = (lambda m: self.dndm(m, z)* m**3* self.u_NFW(k1, m, z)* self.u_NFW(k2, m, z)* self.u_NFW(k3, m, z))

        integral = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        return integral

    def source_source_source_bs_2h(self, k1, k2, k3, z, mmin=1e10, mmax=1e17):
        """2-halo term of Source-Source-Source Bispectrum (not normalized by matterdensity!)

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z) * self.bh(m, z)
        summand1 = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = (lambda m: self.dndm(m, z)* m**2* self.u_NFW(k2, m, z)* self.u_NFW(k3, m, z)* self.bh(m, z) )
        summand1 *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        summand1 *= self.pk_lin(k1, z)

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k3, m, z) * self.bh(m, z)
        summand2 = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = (lambda m: self.dndm(m, z)* m**2* self.u_NFW(k1, m, z)* self.u_NFW(k2, m, z)* self.bh(m, z) )
        summand2 *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        summand2 *= self.pk_lin(k3, z)

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k2, m, z) * self.bh(m, z)
        summand3 = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = (lambda m: self.dndm(m, z)* m**2* self.u_NFW(k1, m, z)* self.u_NFW(k3, m, z)* self.bh(m, z) )
        summand3 *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        summand3 *= self.pk_lin(k2, z)

        return summand1 + summand2 + summand3


    def source_source_source_bs_3h(self, k1, k2, k3, z, mmin=1e10, mmax=1e17):
        """3-halo term of Source-Sourc-Source Bispectrum (not normalized by density!)

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            type (int, optional): Which lens population. Defaults to 1.
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.
        """
        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k1, m, z) * self.bh(m, z)
        integral = np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k2, m, z) * self.bh(m, z)
        integral *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        kernel = lambda m: self.dndm(m, z) * m * self.u_NFW(k3, m, z) * self.bh(m, z)
        integral *= np.trapz(kernel(np.geomspace(mmin, mmax, N)), np.geomspace(mmin, mmax, N))

        return integral * self.bk_lin(k1, k2, k3, z)


    def source_source_source_bs(self, k1, k2, k3, z, mmin=1e10, mmax=1e17):
        """Source-Source-Source Bispectrum 

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
            mmin (float, optional): Minimal halo mass for integral [Msun]. Defaults to 1e10.
            mmax (float, optional): Maximal halo mass for integral [Msun]. Defaults to 1e17.


        Return:
            float: normalized 1-halo term
            float: normalized 2-halo term
            float: normalized 3-halo term
            float: total bispectrum
        """
        rho_bar = ccl.rho_x(self.cosmo, 1 / (1 + z), "matter")

        One_halo = (
            self.source_source_source_bs_1h(k1, k2, k3, z, mmin, mmax)
            / rho_bar
            / rho_bar
            / rho_bar
        )

        Two_halo = (
            self.source_source_source_bs_2h(k1, k2, k3, z, mmin, mmax)
            / rho_bar
            / rho_bar
            / rho_bar
        )

        Three_halo = (
            self.source_source_source_bs_3h(k1, k2, k3, z, mmin, mmax)
            / rho_bar
            / rho_bar
            / rho_bar
        )

        return One_halo, Two_halo, Three_halo, One_halo + Two_halo + Three_halo


    def bk_lin(self, k1, k2, k3, z):
        """ Linear (tree-level) bispectrum

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            k3 (float): wavenumber [1/Mpc]
            z (float): redshift
        """

        c_phi12 = (k3 * k3 - k2 * k2 - k1 * k1) / (2 * k1 * k2)

        c_phi13 = (k2 * k2 - k3 * k3 - k1 * k1) / (2 * k1 * k3)

        c_phi23 = (k1 * k1 - k2 * k2 - k3 * k3) / (2 * k3 * k2)

        return 2 * (
            self.F(k1, k2, c_phi12) * self.pk_lin(k1, z) * self.pk_lin(k2, z)
            + self.F(k1, k3, c_phi13) * self.pk_lin(k1, z) * self.pk_lin(k3, z)
            + self.F(k2, k3, c_phi23) * self.pk_lin(k2, z) * self.pk_lin(k3, z)
        )

    def F(self, k1, k2, cosphi):
        """F-kernel for tree-level bispectrum

        Args:
            k1 (float): wavenumber [1/Mpc]
            k2 (float): wavenumber [1/Mpc]
            cosphi (float): cosine of angle between k1 and k2
        """
        return 0.7143 + 0.2857 * cosphi * cosphi + 0.5 * cosphi * (k1 / k2 + k2 / k2)
