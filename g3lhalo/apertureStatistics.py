import numpy as np
from scipy import integrate


def uHat(eta):
    """
    Computes the aperture statistics filter function \( \hat{u}(\eta) = 0.5 \cdot \eta^2 \cdot \exp(-0.5 \cdot \eta^2) \).
    This is the aperture filter from Crittenden 2002

    Parameters
    ----------
    eta : float or numpy.ndarray
        Input value(s) for the function. Corresponds to theta_ap * ell. dimensionless

    Returns
    -------
    float or numpy.ndarray
        The result of \( \hat{u}(\eta) \) computed for the given `eta`.

    """
    tmp = 0.5 * eta**2
    return tmp * np.exp(-tmp)


def custom_interpolate(ell1_values, ell2_values, phi_values, bkappa_grid, ell1, ell2, phi):
    """
    Performs trilinear interpolation on a 3D grid of `bkappa` values based on the given 
    `ell1`, `ell2`, and `phi` coordinates. 

    Parameters
    ----------
    ell1_values : numpy.ndarray
        1D array of `ell1` coordinate values in ascending order.
    ell2_values : numpy.ndarray
        1D array of `ell2` coordinate values in ascending order.
    phi_values : numpy.ndarray
        1D array of `phi` coordinate values in ascending order.
    bkappa_grid : numpy.ndarray
        3D array containing `bkappa` values, where each entry corresponds to a specific 
        combination of `ell1`, `ell2`, and `phi` values.
    ell1 : float
        Target `ell1` coordinate for interpolation.
    ell2 : float
        Target `ell2` coordinate for interpolation.
    phi : float
        Target `phi` coordinate for interpolation.

    Returns
    -------
    float
        Interpolated `bkappa` value at the specified (`ell1`, `ell2`, `phi`) point.

    Notes
    -----
    This function assumes that `ell1_values`, `ell2_values`, and `phi_values` are 
    uniformly spaced and ordered in ascending fashion. The method will perform 
    boundary checks and handle cases where the coordinates fall at the edges of 
    the grid.
    """
    i = np.searchsorted(ell1_values, ell1) - 1
    j = np.searchsorted(ell2_values, ell2) - 1
    k = np.searchsorted(phi_values, phi) - 1
    
    # Handle boundaries
    i = max(min(i, len(ell1_values) - 2), 0)
    j = max(min(j, len(ell2_values) - 2), 0)
    k = max(min(k, len(phi_values) - 2), 0)
    
    x1, x2 = ell1_values[i], ell1_values[i+1]
    y1, y2 = ell2_values[j], ell2_values[j+1]
    z1, z2 = phi_values[k], phi_values[k+1]
    
    # Trilinear interpolation
    f111 = bkappa_grid[i, j, k]
    f112 = bkappa_grid[i, j, k+1]
    f121 = bkappa_grid[i, j+1, k]
    f122 = bkappa_grid[i, j+1, k+1]
    f211 = bkappa_grid[i+1, j, k]
    f212 = bkappa_grid[i+1, j, k+1]
    f221 = bkappa_grid[i+1, j+1, k]
    f222 = bkappa_grid[i+1, j+1, k+1]
    
    f11 = (f111 * (z2 - phi) + f112 * (phi - z1)) / (z2 - z1)
    f12 = (f121 * (z2 - phi) + f122 * (phi - z1)) / (z2 - z1)
    f21 = (f211 * (z2 - phi) + f212 * (phi - z1)) / (z2 - z1)
    f22 = (f221 * (z2 - phi) + f222 * (phi - z1)) / (z2 - z1)
    
    f1 = (f11 * (y2 - ell2) + f12 * (ell2 - y1)) / (y2 - y1)
    f2 = (f21 * (y2 - ell2) + f22 * (ell2 - y1)) / (y2 - y1)
    
    return (f1 * (x2 - ell1) + f2 * (ell1 - x1)) / (x2 - x1)


def custom_interpolate_1d(x_values, y_values, x):
    """
    Performs linear interpolation on a 1D grid of `y` values based on the given `x` coordinate.

    Parameters
    ----------
    x_values : numpy.ndarray
        1D array of `x` coordinate values in ascending order.
    y_values : numpy.ndarray
        1D array of `y` values corresponding to each `x` coordinate in `x_values`.
    x : float
        Target `x` coordinate for interpolation.

    Returns
    -------
    float
        Interpolated `y` value at the specified `x` point.

    Notes
    -----
    This function uses linear interpolation to estimate the `y` value at the given `x` coordinate.
    The function assumes `x_values` are in ascending order. Boundary cases are handled by clamping
    the interpolation interval to the edges of the `x_values` array if `x` is outside the range.
    """
    # Find the interval in x_values that contains x
    i = np.searchsorted(x_values, x) - 1

    # Handle boundaries: make sure i is within the valid range
    i = max(min(i, len(x_values) - 2), 0)
    
    # Get the x-values and y-values at the interval boundaries
    x1, x2 = x_values[i], x_values[i+1]
    y1, y2 = y_values[i], y_values[i+1]

    # Linear interpolation formula
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def integrand_third_order_helper(ell1, ell2, phi, theta1, theta2, theta3, bispec, ell1_values, ell2_values, phi_values):

    ell3 = np.sqrt(ell1**2 + ell2**2 + 2 * ell1 * ell2 * np.cos(phi))

    # Interpolate bkappa
    bk=custom_interpolate(ell1_values, ell2_values, phi_values, bispec, ell1, ell2, phi)

    return ell1 * ell2 * bk * uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3)


def integrand_second_order_helper(ell, theta, pkappas, ell_values):
    # Interpolate bkappa
    pk = custom_interpolate_1d(ell_values, pkappas, ell)    

    return ell * pk * uHat(ell * theta) **2

class apertureStatistics:

    def __init__(self, powerspec=None, bispec=None) -> None:
        
        if powerspec!=None:
            self.powerspec=powerspec['ps']
            self.ell_values=powerspec['ell']

        if bispec!=None:
            self.bispec=bispec['bs']
            self.ell1_values=bispec['ell1']
            self.ell2_values=bispec['ell2']
            self.phi_values=bispec['phi']

    
    def integrand_third_order(self, ell1, ell2, phi, theta1, theta2, theta3):
        return integrand_third_order_helper(ell1, ell2, phi, theta1, theta2, theta3, self.bispec,
                                            self.ell1_values, self.ell2_values, self.phi_values)
    
    def third_order(self, theta1, theta2, theta3):
        lMax=10./np.min([theta1, theta2, theta3])
        ell1_limits=[1, lMax]
        ell2_limits=[1, lMax]
        phi_limits=[0, 2*np.pi]

        def integrand_wrapper(ell1, ell2, phi):
            return self.integrand_third_order(ell1, ell2, phi, theta1, theta2, theta3)
        
        result, error=integrate.nquad(integrand_wrapper, [ell1_limits, ell2_limits, phi_limits], opts={'epsrel': 1e-4, 'limit': 15})

        prefactor=1/8/np.pi**3
        return result*prefactor



    def integrand_second_order(self, ell, theta):
        return integrand_second_order_helper(ell, theta, self.powerspec, self.ell_values)
    

    def second_order(self, theta):
        lMax=10./theta
        ell_limits=[1, lMax]

        def integrand_wrapper(ell):
            return self.integrand_second_order(ell, theta)
        
        result, error=integrate.nquad(integrand_wrapper, [ell_limits], opts={'epsrel': 1e-4})

        prefactor=1./2./np.pi

        return result*prefactor
