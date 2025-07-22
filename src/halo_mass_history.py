import numpy as np
from scipy.optimize import brentq

class HaloMassHistory:
    """ Class to calculate the evolution of halo masses over time given the progenitor halo mass at some redshift """
    def __init__(self, mass_at_zi, zi, halomassfunction):
        """
        Parameters:
        -----------
        mass_at_zi : Halo mass at the progenitor redshift zi.
        zi : Progenitor redshift (where the mass is known).
        halomassfunction : object
            Instance of the HaloMassFunction class (contains methods for D(z), sigma(M), etc.)
        
        Based on arXiv:1409.5228
        """
        self.Mzi = mass_at_zi  # Progenitor mass at redshift zi
        self.zi = zi           # Progenitor redshift
        self.halomassfunction = halomassfunction
        
        # Use Brent's method to determine the descendant mass at z=0 (M0) such that:
        #    Mzi = M0 * (1+zi)^alpha(M0) * exp(beta(M0)*zi).
        self.M0 = self.find_M0_brentq(self.Mzi, self.zi)
    
    def growth_factor_derivative_at_z0(self):
        """Approximate derivative of the growth factor at z=0."""
        z0 = 0.0
        delta_z = 1e-2
        D_z0 = self.halomassfunction.overden.Dofz(z0)
        D_z0_plus = self.halomassfunction.overden.Dofz(z0 + delta_z)
        return (D_z0_plus - D_z0) / delta_z
    
    def compute_s(self, mass):
        """Compute the variance S(M) for a given mass using sigma^2."""
        sigma = self.halomassfunction.overden.sigmaof_M_z(mass)
        return sigma**2
    
    def compute_q(self, M):
        """Compute q as defined in the EPS equations."""
        log_M = np.log10(M)
        zf = -0.0064 * log_M**2 + 0.0237 * log_M + 1.8837
        return 4.137 * zf**(-0.9476)
    
    def compute_f(self, M):
        """Compute f(M) = 1/sqrt[S(M/q) - S(M)]."""
        q = self.compute_q(M)
        S_Mq = self.compute_s(M / q)
        S_M = self.compute_s(M)
        return 1.0 / np.sqrt(S_Mq - S_M)
    
    def compute_alpha_and_beta(self, M):
        """
        Compute the EPS parameters alpha and beta from a given mass M.
        Here:
          beta = -f(M)
          alpha = [1.686*sqrt(2/pi)*(dD/dz at z=0) + 1] * f(M)
        """
        f_val = self.compute_f(M)
        dDdz = self.growth_factor_derivative_at_z0()
        beta = -f_val
        alpha = (1.686 * np.sqrt(2.0/np.pi) * dDdz + 1) * f_val
        return alpha, beta
    
    def func_to_solve(self, M0):
        """
        Define the function whose root (in M0) ensures that the descendant mass, when
        extrapolated back to z=zi, recovers the known progenitor mass Mzi:
          f(M0) = M0 * (1+zi)^alpha(M0) * exp(beta(M0)*zi) - Mzi = 0.
        """
        alpha, beta = self.compute_alpha_and_beta(M0)
        return M0 * (1 + self.zi)**alpha * np.exp(beta * self.zi) - self.Mzi
    
    def find_M0_brentq(self, Mzi, zi, tol=1e-4):
        """
        Use Brent's method to solve for M0 such that:
          M0*(1+zi)^alpha(M0)*exp(beta(M0)*zi) = Mzi.
        """
        lower_bound = Mzi 
        upper_bound = Mzi * 1.1e4  # Adjust this bracket if needed but I find this to be the best value
        f_low = self.func_to_solve(lower_bound)
        f_high = self.func_to_solve(upper_bound)

        # max_factor = 1e5 
        # factor = 9e3
        # while f_low * f_high > 0 and factor < max_factor:
        #     factor *= 1.2
        #     upper_bound = Mzi * factor
        #     f_high = self.func_to_solve(upper_bound)

        if f_low * f_high > 0:
            raise ValueError("No sign change found in the bracket for M0.")
        M0_root = brentq(self.func_to_solve, lower_bound, upper_bound, xtol=tol)
        return M0_root
    
    def mass_at_z_from_zi(self, z):
        """
        Calculate the descendant halo mass M(z) at redshift z (z may be < or > zi)
        using the EPS-based formula:
           M(z) = M0 * (1+z)^alpha * exp(beta*z)
        where M0, alpha, and beta are determined from the Brent search.
        """
        alpha, beta = self.compute_alpha_and_beta(self.M0)
        return self.M0 * (1 + np.array(z))**alpha * np.exp(beta * np.array(z))
