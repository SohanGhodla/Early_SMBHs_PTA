import numpy as np
from scipy.integrate import quad
from scipy.integrate import quad_vec
from scipy.stats import lognorm
from joblib import Parallel, delayed
# from astropy.cosmology import LambdaCDM, z_at_value
import astropy.units as u
from scipy.optimize import brentq
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, RegularGridInterpolator
import matplotlib.pyplot as plt

from src.cosmolopy import constants
from src.halo_mass_function import HaloMassFunction
from src.halo_merger_rate import HaloMergerRateDensity


class BHMergerRateDensityTrad:
    """
    Class to calculate the comoving black hole merger rate density R_BH using the Extended Press-Schechter (EPS) formalism.
    The merger rate is computed as d² R_BH / (d m1 d m2) with units of Mpc^{-3} yr^{-1} Msun^{-2}.
    """
    def __init__(self, z_merger, m1, m2, t_i = 6e8):
        """
        Parameters:
        - z_merger: Redshift at which BHMergerRateDensity is evaluated
        - t_i: Initial time for integration (yr), default set to 1 Myr.
        """
        self.z_merger = z_merger
        self.t_i = t_i  # Initial time in yr
        self.h = 0.674  # Hubble parameter 
        self.H0 = self.h * 100 / constants.Mpc_km * constants.yr_s # Hubble constant yr^-1
        self.omega_m0 = 0.315
        self.omega_l0 = 0.685

        # Define grids
        self.n_z = 300
        self.z_grid = np.logspace(-2.1, 1, self.n_z)
        self.M_h_grid = np.logspace(2, 20, 2000)
        self.interpolators = []
        self.precompute_M_of_m()

        # To precompute the Rh values (saves time)
        if not hasattr(self, 'Rh_interpolator'):
            self.precompute_Rh_interpolator(m1, m2)

        # Precompute the p_merg interpolation for delay times (only once)
        if not hasattr(self, 'p_merg_interp'):  # Check if interpolation already exists
            self.p_merg()  # Precompute once and store it for later use
        

    def stellar_mass_from_halo_mass(self, M_h, z):
        """ return stellar mass given a halo mass
         Based on arXiv:2001.02230 """
        # Parameters for SHMR evolution
        B = 11.79
        mu = 0.20
        C = 0.046
        nu = -0.38
        D = 0.709
        eta = -0.18
        F = 0.043
        E = 0.96

        # Compute redshift-dependent parameters
        M_A = 10**(B + z * mu)
        A = C * (1 + z)**nu
        gamma = D * (1 + z)**eta
        beta = F * z + E

        # SHMR equation:
        #   M_star = M_h * 2 * A * [ (M_h/M_A)^(-beta) + (M_h/M_A)^(gamma) ]^(-1)
        M_star = M_h * 2 * A * (((M_h / M_A)**(-beta) + (M_h / M_A)**(gamma))**(-1))
        return M_star

    def black_hole_mass_from_stellar_mass(self, M_star):
        """ Return black hole mass for a given stellar mass.
        Based on arXiv:1508.06274 
        """
        return 10**(8.95 + 1.40 * np.log10(M_star / 1e11))
    
    def precompute_M_of_m(self):
        """For a grid of redshifts, precompute an interpolator that maps
           BH mass -> halo mass."""
        
        # For each redshift, compute the corresponding BH masses for our halo mass grid.
        for z in self.z_grid:
            # For each halo mass in M_h_grid, compute the stellar mass and then BH mass.
            M_star = self.stellar_mass_from_halo_mass(self.M_h_grid, z)
            bh_masses = self.black_hole_mass_from_stellar_mass(M_star)
            
            # Build a 1D interpolator that inverts the relation: input bh_mass, output halo mass.
            interp_func = InterpolatedUnivariateSpline(bh_masses, self.M_h_grid, k=3)
            self.interpolators.append(interp_func)

    def M_of_m(self, m, z):
        """Return the halo mass for a given BH mass array `m` and redshift `z`.
        Here, m can be a 2D meshgrid; we flatten it for interpolation and then reshape."""
        m_flat = m.flatten()
        
        # If z is out of bounds, simply use the nearest available interpolator.
        if z <= self.z_grid[0]:
            M_h_flat = self.interpolators[0](m_flat)
        elif z >= self.z_grid[-1]:
            M_h_flat = self.interpolators[-1](m_flat)
        else:
            # Find indices in z_grid that bracket the desired z.
            idx = np.searchsorted(self.z_grid, z)
            z_low = self.z_grid[idx - 1]
            z_high = self.z_grid[idx]
            interp_low = self.interpolators[idx - 1]
            interp_high = self.interpolators[idx]
        
            # Evaluate the interpolators on the flattened array.
            M_h_low = interp_low(m_flat)
            M_h_high = interp_high(m_flat)
        
            # Linear interpolation in redshift.
            weight = (z - z_low) / (z_high - z_low)
            M_h_flat = M_h_low + weight * (M_h_high - M_h_low)
        
        # Reshape the interpolated halo masses back to the original meshgrid shape.
        return M_h_flat.reshape(m.shape)


    def dM_dm(self, m, M, z, delta=1e-3):
        """
        Numerically compute dM/dm at a given m (which can be an array) and redshift z
        using central finite differences.
        
        Parameters:
            m : The black hole mass (can be 1D or 2D meshgrid).
            z : Redshift
            delta : The small step size for finite differences.
        
        Returns:
            dM_dm : The numerical derivative dM/dm with the same shape as m.
        """
        # Can also use spline derivative
      
        M_plus = self.M_of_m(m + delta, z)
        M_minus = self.M_of_m(m - delta, z)
        
        # Central difference derivative
        dM_dm_val = (M_plus - M_minus) / (2 * delta)
        
        return dM_dm_val

    
    def p_occ(self, m, z_prime):
        """
        Occupation fraction: Returns the probability that a halo at redshift z_prime has a BH.
        """
        return 1

    # def p_merg(self, tau0=1.0, sigma=0.5):
    #     """
    #     Returns a 1/yr PDF for the delay-time τ (in Gyr) between halo merger
    #     and BH merger, assuming a log-normal distribution whose *mean* is tau0.

    #     Parameters:
    #     -----------
    #     tau0 : float
    #         Mean merger delay time [Gyr].
    #     sigma : float
    #         Standard deviation of ln(τ) (i.e. log-space scatter).
        
    #     Objective:
    #         creates self.p_merg_interp(τ) with p(τ) in [1/yr].
    #     """

    #     # numerical grid parameters
    #     tau_min = 1e-4           # avoid τ=0
    #     tau_max = 14.0           # upper end for grid (can be extended)
    #     num_pts = 1000

    #     # tau-grid (in Gyr)
    #     tau_grid = np.linspace(tau_min, tau_max, num_pts)

    #     # Compute the log-normal mu so that mean = tau0:
    #     #    E[tau] = exp(mu + 0.5*sigma^2) = tau0  ->  mu = ln(tau0) - 0.5*sigma^2
    #     mu = np.log(tau0) - 0.5 * sigma**2

    #     # evaluate the log-normal PDF [per Gyr]
    #     pdf_per_gyr = lognorm.pdf(
    #         tau_grid,
    #         s=sigma,
    #         scale=np.exp(mu),
    #     )

    #     # Convert to per-year
    #     pdf_per_year = pdf_per_gyr / 1e9

    #     # Interpolator
    #     self.p_merg_interp = interp1d(
    #         tau_grid,
    #         pdf_per_year,
    #         kind='linear',
    #         bounds_error=False,
    #         fill_value=0.0, # zero outside the grid
    #     )

    def p_merg(self, tau0=1.0, sigma=1/3):
        """
        Returns a 1/yr PDF for the delay-time τ (in Gyr) between halo merger
        and BH merger, assuming a log-normal distribution whose *mean* is tau0.

        Parameters:
        -----------
        tau0 : float
            Mean merger delay time [Gyr].
        sigma : float
            Standard deviation of ln(τ) (i.e. log-space scatter).
        
        Objective:
            creates self.p_merg_interp(τ) with p(τ) in [1/yr].
        """

        # numerical grid parameters
        tau_min = 1e-4           # avoid τ=0
        tau_max = 14.0           # upper end for grid (can be extended)
        num_pts = 1000

        # tau-grid (in Gyr)
        tau_grid = np.linspace(tau_min, tau_max, num_pts)

        # Compute the log-normal mu so that mean = tau0:
        #    E[tau] = exp(mu + 0.5*sigma^2) = tau0  ->  mu = ln(tau0) - 0.5*sigma^2
        mu = np.log(tau0) - 0.5 * sigma**2

        # evaluate the log-normal PDF [per Gyr]
        pdf_per_gyr = lognorm.pdf(
            tau_grid,
            s=sigma,
            scale=np.exp(mu),
        )

        # Convert to per-year
        pdf_per_year = pdf_per_gyr / 1e9

        # Interpolator
        self.p_merg_interp = interp1d(
            tau_grid,
            pdf_per_year,
            kind='linear',
            bounds_error=False,
            fill_value=0.0, # zero outside the grid
        )

    def cosmic_time(self, z):
        """
        Calculate cosmic time [in years] at redshift z.
        """
        # Function E(z)
        def E(zp):
            return np.sqrt(self.omega_l0 +
            (1.0 - self.omega_l0 - self.omega_m0) * (1.0 + zp)**2 +
            self.omega_m0 * (1.0 + zp)**3 )
        
        # Integrate to compute cosmic time
        integrand = lambda zp: 1 / ((1 + zp) * E(zp))
        integral, error = quad(integrand, z, np.inf) # # Integrate from z to a very high redshift ~ infinity
        t = integral / self.H0  # Cosmic time years
        return t  # in yrs
    

    def redshift_from_time(self, t_prime):
        """
        Numerically invert cosmic time to find z_prime corresponding to t_prime (t_prime in years)
        """
        # Define the function whose root we want to find
        def f(z):
            return self.cosmic_time(z) - t_prime

        # Set bounds for redshift
        z_min = 0.0
        z_max = 200.0 

        # Use brentq to find the root
        z_prime = brentq(f, z_min, z_max, xtol=1e-5)
        # print("t_prime =", t_prime, "cosmic_time(0) =", self.cosmic_time(0), "cosmic_time(120) =", self.cosmic_time(120))

        return z_prime

    def compute_Rh_at_z(self, z_prime, m1, m2):
        """ Helper function for precompute_Rh_interpolator function"""
        # Create a halo mass function instance at this redshift.
        hmf_z = HaloMassFunction(z_prime)
        # Initialize the halo merger rate density class with this hmf.
        hmr_z = HaloMergerRateDensity(hmf_z)
        # Convert black hole masses to halo masses.
        M1 = self.M_of_m(m1, z_prime)
        M2 = self.M_of_m(m2, z_prime)
        # Compute the halo merger rate.
        Rh = hmr_z.compute_Rh(M1 / self.h, M2 / self.h)
        return Rh


    def precompute_Rh_interpolator(self, m1, m2):
        """" To speed up, precompute the halo merger rate density Rh
        Parameters: 
            the mass of the black holes m1, m2
        Returns: 
            Rh_interpolator (interploted over a range of redshifts) """
        
        n_z = 300
        z_prime_grid = np.logspace(-2.1, 1, n_z)  
        # N1, N2 = m1.shape
        
        # Compute in parallel:
        Rh_values_list = Parallel(n_jobs=-1)(
            delayed(self.compute_Rh_at_z)(z_prime, m1, m2) for z_prime in z_prime_grid
        )
        Rh_values = np.array(Rh_values_list)  # Shape: (n_z, N1, N2)

        # Reshape for interpolation:
        Rh_values_reshaped = Rh_values.reshape(n_z, -1)
        
        # Build the interpolator:
        self.Rh_interpolator = RegularGridInterpolator(
            (z_prime_grid,), Rh_values_reshaped, bounds_error=False, fill_value=0
        )

    def compute_R_BH(self, m1, m2, t):
        """
        Compute the comoving black hole merger rate density d²R_BH / (d m1 d m2).

        Parameters:
        - m1: Mass of the first black hole (Msun).
        - m2: Mass of the second black hole (Msun).
        - t: Current cosmic time (yr).

        Returns:
        - R_BH: The comoving black hole merger rate density in units of Mpc^{-3} yr^{-1} Msun^{-2}.
        """

        # Integrand as a function of t'
        def integrand(t_prime):
            
            z_prime = self.redshift_from_time(t_prime)
            # z_prime = z_at_value(self.cosmo.age, (t_prime / 1e9) * u.Gyr).value
            # print('z_prime =', z_prime, t_prime/1e9)
            
            # ------------------------------------
            # ----- Uncomment if want tau \neq 0  ------
            # ------------------------------------
            # Compute R_h(t') = d^3 R_h / dM1 dM2 dt' at z_prime
            Rh_flat = self.Rh_interpolator(np.array([z_prime]))[0]
            Rh = Rh_flat.reshape(m1.shape)

            # # ------------------------------------
            # # ------ uncomment if want tau = 0  ------
            # # ------------------------------------
            # hmf_z = HaloMassFunction(z_prime)
            # hmr_z = HaloMergerRateDensity(hmf_z)
            # # Compute R_h(t') = d^3 R_h / dM1 dM2 dt'
            # M1 = self.M_of_m(m1, z_prime)
            # M2 = self.M_of_m(m2, z_prime)
            # Rh = hmr_z.compute_Rh(M1 / self.h, M2 / self.h)  # Units:  yr^{-1} Msun^{-2} Mpc^{-3}


            # Compute occupation probabilities at z
            p_occ_m1 = 1 #self.p_occ(m1, z_prime)
            p_occ_m2 = 1 #self.p_occ(m2, z_prime)

            # Compute delay time tau = t - t'
            tau = t - t_prime  # in years
            tau_Gyr = tau / 1e9  # Convert years to Gyr

            # Compute merger probability with delay time
            p_merger = self.p_merg_interp(tau_Gyr) 

            M1 = self.M_of_m(m1, z_prime)
            M2 = self.M_of_m(m2, z_prime)

            # Compute dM1/dm1 and dM2/dm2
            dM1_dm1 = self.dM_dm(m1, M1, z_prime)  # dM/dm
            dM2_dm2 = self.dM_dm(m2, M2, z_prime)  # dM/dm

            integrand_value = p_occ_m1 * p_occ_m2 * p_merger * dM1_dm1 * dM2_dm2 * Rh 

            return integrand_value

        # Perform the integration from t_i to t (where t is the time of merger of black holes)
        if t < self.t_i:
            raise ValueError("t_merger [i.e., t] is less than t_i")

        # ------------------------------------
        # ----- Uncomment if want tau \neq 0  ------
        # ------------------------------------
        # Perform the integration from t_i to t 
        result, _ = quad_vec(integrand, self.t_i, t, limit = 25) # increase the limit of get errors/warinigs

        # ------------------------------------
        # ----- Uncomment if want tau = 0 (i.e., instant merger)  ------
        # ------------------------------------
        # result = integrand(t)

        return result  # Units:  Mpc^{-3} yr^{-1} Msun^{-2}
    



# This is the old BH-halo mass relation
 # def M_of_m(self, m, hmf_z, z):
    #     """
    #     Black hole mass as a function of halo mass using the prescription in Loeb and Whyte 2004.
        
    #     Parameters:
    #     - m: Black hole mass [M_sun]
        
    #     Returns:
    #     - M: Halo mass [M_sun]
    #     """
    #     # Define parameters
    #     epsilon_0 = 10**8 # Describes Slope
    #     gamma = 5. #3.0    # Describes Slope
        
    #     # Compute Omega_M(z)
    #     Omega_M_z = hmf_z.overden.omega_matter_of_z(z)
        
    #     # Compute Delta_c(z) using the approximation Delta_c = 18π² + 82x - 39x²
    #     x = Omega_M_z - 1
    #     Delta_c = 18 * np.pi**2 + 82 * x - 39 * x**2
        
    #     # Compute the halo mass M 
    #     term1 = (m / epsilon_0) ** (3 / gamma)
    #     term2 = (hmf_z.omega_m0 / Omega_M_z * Delta_c / (18 * np.pi**2)) ** (-1 / 2)
    #     term3 = 10.5 #self.h ** (-1)
    #     term4 = (1 + z) ** (-3 / 2)
        
    #     M = 1e12 * term1 * term2 * term3 * term4  # M in M_sun
    #     return M


    # def dM_dm(self, m, hmf_z, z):
    #     """
    #     Derivative dM/dm based on m(M) relation.
        
    #     Parameters:
    #     - m: Black hole mass [M_sun]
        
    #     Returns:
    #     - dM/dm: Derivative [M_sun/M_sun = unitless]
    #     """
    #     gamma = 5 
    #     M = self.M_of_m( m, hmf_z, z)
    #     dM_dm = 3/gamma * (M / m)
        
    #     return dM_dm
    