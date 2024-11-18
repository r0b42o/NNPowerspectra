import numpy as np
import pyccl as ccl
from scipy import interpolate

c_over_H0 = 4400

class projectedSpectra:

    def __init__(self, halomod, zmin=0, zmax=2, nzbins=25):
        """Limber projection initialization.

        Args:
            halomod (g3lhalo.halomodel): Halomodel, needs to be fully initialized
            zmin (float, optional): Minimal redshift for redshift binning. Defaults to 0.
            zmax (float, optional): Maximal redshift for redshift binning. Defaults to 2.
            nzbins (int, optional): Number of redshift bins. Defaults to 25.
        """
        self.cosmo=halomod.cosmo
        self.halomod=halomod
        self.z_bins=np.linspace(zmin, zmax, nzbins)
        self.zs=0.5*(self.z_bins[1:]+self.z_bins[:1])
        self.deltaZs=self.z_bins[1:]-self.z_bins[:1]

        self.ws=ccl.comoving_radial_distance(self.cosmo, 1/(1+self.zs))

        self.dwdz = 1 / np.sqrt(ccl.h_over_h0(self.cosmo, 1/(1+self.zs))) * c_over_H0

    def set_nz_lenses(self, zs, n):
        """Sets and interpolates lens redshift distribution (which needs to be normalized)

        Args:
            zs (np.array): redshift range
            n (np.array): n(z)
        """
        self.n_l = interpolate.interp1d(zs, n, fill_value=0, bounds_error=False)

    def set_nz_sources(self, zs, n):
        """Sets and interpolates source redshift distribution (which needs to be normalized)
        and calculates lensing efficiency

        Args:
            zs (np.array): redshift range
            n (np.array): n(z)
        """
        self.n_s = interpolate.interp1d(zs, n, fill_value=0, bounds_error=False)
        self.set_lensing_efficiency()

    def set_lensing_efficiency(self):
        """ Calculates lensing efficiency for given source redshift distribution
        """
        n_redshift_bins=len(self.zs)
        dz=(self.zs[-1]-self.zs[0])/n_redshift_bins

        self.gs = np.zeros_like(self.zs)

        nz=self.n_s(self.zs)

        w_inv=1.0/self.ws

        for j in range(n_redshift_bins):
            w_diff=self.ws[j:] - self.ws[j]
            nz_znow = nz[j:]

            integrand= nz_znow * w_diff * w_inv[j:]
            trapezoidal_sum = np.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])

            self.gs[j] = trapezoidal_sum*dz

        self.gs=np.nan_to_num(self.gs)

        
        Om=self.cosmo._params['Omega_c']+self.cosmo._params['Omega_b']
        self.gs*=3/2*Om/c_over_H0/c_over_H0*(1+self.zs)

    def C_kk(self, ell):
        """ Calculates matter-matter C(ell)

        Args:
            ell (float): wave number

        Returns:
            float: 1_halo term
            float: 2-halo term
            float: Total C(ell)
        """
        ell=np.array(ell)
        result_1h = np.zeros_like(ell)
        result_2h = np.zeros_like(ell)

        for i, z in enumerate(self.zs):
            P1h, P2h, P_all=self.halomod.source_source_ps(ell/self.ws[i], z)

            result_1h+=self.gs[i]**2*P1h*self.deltaZs[i] * self.dwdz[i]
            result_2h+=self.gs[i]**2*P2h*self.deltaZs[i] * self.dwdz[i]

        return result_1h, result_2h, result_1h+result_2h

    def C_kg(self, ell, type=1):
        """ Calculates matter-galaxy C(ell)

        Args:
            ell (float): wave number
            type (int, optional): Which lens population. Defaults to 1.

        Returns:
            float: 1_halo term
            float: 2-halo term
            float: Total C(ell)
        """
        ell=np.array(ell)
        result_1h = np.zeros_like(ell)
        result_2h = np.zeros_like(ell)

        for i, z in enumerate(self.zs):
            P1h, P2h, P_all=self.halomod.source_lens_ps(ell/self.ws[i], z, type)

            result_1h+=self.gs[i]/self.ws[i]*self.n_l(z)*P1h*self.deltaZs[i] #/ self.dwdz[i]
            result_2h+=self.gs[i]/self.ws[i]*self.n_l(z)*P2h*self.deltaZs[i] #/ self.dwdz[i]

        return result_1h, result_2h, result_1h+result_2h
    

    def C_gg(self, ell, type1=1, type2=1):
        """ Calculates galaxy-galaxy C(ell)

        Args:
            ell (float): wave number
            type1 (int, optional): Which first lens population. Defaults to 1.
            type2 (int, optional): Which second lens population. Defaults to 1.


        Returns:
            float: 1_halo term
            float: 2-halo term
            float: Total C(ell)
        """
        ell=np.array(ell)
        result_1h = np.zeros_like(ell)
        result_2h = np.zeros_like(ell)

        for i, z in enumerate(self.zs):
            P1h, P2h, P_all=self.halomod.lens_lens_ps(ell/self.ws[i], z, type1, type2)

            result_1h+=self.n_l(z)**2/self.ws[i]**2*P1h*self.deltaZs[i] / self.dwdz[i]
            result_2h+=self.n_l(z)**2/self.ws[i]**2*P2h*self.deltaZs[i] / self.dwdz[i]

        return result_1h, result_2h, result_1h+result_2h

    
    def C_kgg(self, ell1, ell2, ell3, type1=1, type2=1):
        """ Calculates matter-galaxy-galaxy C(ell)

        Args:
            ell1 (float): wave number
            ell2 (float): wave number
            ell3 (float): wave number
            type1 (int, optional): Which first lens population. Defaults to 1.
            type2 (int, optional): Which second lens population. Defaults to 1.


        Returns:
            float: 1_halo term
            float: 2-halo term
            float: 3-halo term
            float: Total C(ell_1, ell_2, ell_3)
        """
        result_1h=0
        result_2h=0
        result_3h=0

        for i, z in enumerate(self.zs):
            w=self.ws[i]
            P1h, P2h, P3h, _ = self.halomod.source_lens_lens_bs(ell1/w, ell2/w, ell3/w, z, type1, type2)

            result_1h+=self.gs[i]*self.n_l(z)**2/w**3*P1h*self.deltaZs[i] / self.dwdz[i]
            result_2h+=self.gs[i]*self.n_l(z)**2/w**3*P2h*self.deltaZs[i] / self.dwdz[i]
            result_3h+=self.gs[i]*self.n_l(z)**2/w**3*P3h*self.deltaZs[i] / self.dwdz[i]
        return result_1h, result_2h, result_3h, result_1h+result_2h+result_3h


    def C_kkg(self, ell1, ell2, ell3, type=1):
        """ Calculates matter-matter-galaxy C(ell)

        Args:
            ell1 (float): wave number
            ell2 (float): wave number
            ell3 (float): wave number
            type (int, optional): Which lens population. Defaults to 1.

        Returns:
            float: 1_halo term
            float: 2-halo term
            float: 3-halo term
            float: Total C(ell_1, ell_2, ell_3)
        """
        result_1h=0
        result_2h=0
        result_3h=0

        for i, z in enumerate(self.zs):
            w=self.ws[i]
            B1h, B2h, B3h, _ = self.halomod.source_source_lens_bs(ell1/w, ell2/w, ell3/w, z, type)

            result_1h+=self.gs[i]**2*self.n_l(z)/w**2*B1h*self.deltaZs[i] #/ self.dwdz[i]
            result_2h+=self.gs[i]**2*self.n_l(z)**2/w**2*B2h*self.deltaZs[i] #/ self.dwdz[i]
            result_3h+=self.gs[i]**2*self.n_l(z)**2/w**2*B3h*self.deltaZs[i] #/ self.dwdz[i]
        return result_1h, result_2h, result_3h, result_1h+result_2h+result_3h
    

    def C_kkk(self, ell1, ell2, ell3, type=1):
        """ Calculates matter-matter-matter C(ell)

        Args:
            ell1 (float): wave number
            ell2 (float): wave number
            ell3 (float): wave number
            type (int, optional): Which lens population. Defaults to 1.

        Returns:
            float: 1_halo term
            float: 2-halo term
            float: 3-halo term
            float: Total C(ell_1, ell_2, ell_3)
        """
        result_1h=0
        result_2h=0
        result_3h=0

        for i, z in enumerate(self.zs):
            w=self.ws[i]
            B1h, B2h, B3h, _ = self.halomod.source_source_source_bs(ell1/w, ell2/w, ell3/w, z)

            result_1h+=self.gs[i]**3/w*B1h*self.deltaZs[i] * self.dwdz[i]
            result_2h+=self.gs[i]**3/w*B2h*self.deltaZs[i] * self.dwdz[i]
            result_3h+=self.gs[i]**3/w*B3h*self.deltaZs[i] * self.dwdz[i]
        return result_1h, result_2h, result_3h, result_1h+result_2h+result_3h
    