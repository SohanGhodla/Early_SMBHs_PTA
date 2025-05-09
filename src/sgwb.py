from src.cosmolopy import constants
from src.bh_merger_rate import BHMergerRateDensity

import numpy as np
from scipy.integrate import quad
from joblib import Parallel, delayed
from scipy.interpolate import interp1d


class SGWBEnergyDensity:
    """
    Class to calculate the energy density of the stochastic gravitational wave background (SGWB)
    from merging (super)massive black hole binaries.
    """
    def __init__(self, m1, m2, mass1, mass2):
        """
        Initialize the SGWBEnergyDensity class.
        - m1, m2: Black hole masses (meshgrids) [Msun]
        - mass1, mass2: Arrays of black hole masses [Msun]
        """
    
        self.h = 0.674
        self.H0 = self.h * 100 / constants.Mpc_km  # in seconds
        self.rho_c = 3 * (self.H0)**2 / (8*np.pi*constants.G_const_Mpc_Msun_s) 
        self.omega_m =  0.315
        self.omega_l = 0.685
        self.c_light = constants.c_light_Mpc_s  # Speed of light [Mpc/s]
        self.G_Mpc_Msun_s = constants.G_const_Mpc_Msun_s # Gravitational const.
        self.m1 = m1
        self.m2 = m2
        self.mass1 = mass1
        self.mass2 = mass2

        # Polynomial coefficients from table 1 in Ajith et al. 2009 (https://arxiv.org/pdf/0710.2335)
        self.polynomial_coeffs = {
            'f_merg_coeff': (2.9740e-1, 4.4810e-2, 9.5560e-2),  
            'f_ring_coeff': (5.9411e-1, 8.9794e-2, 1.9111e-1),  
            'f_cut_coeff' : (8.4845e-1, 1.2848e-1, 2.7299e-1),   
            'sigma_coeff'        : (5.0801e-1, 7.7515e-2, 2.2369e-2)  
        }

        self.precompute_comoving_distance(z_min=0.0, z_max=10.0, num_points=1000)
        self.precompute_dV(z_min=0.0, z_max=10.0, num_points=1000)
        self.precompute_H_z(z_min=0.0, z_max=10.0, num_points=1000)
        self.precompute_cosmic_time(z_min=0.0, z_max=10.0, num_points=1000)

        self.precompute_Delta_c(z_min=0.0, z_max=10.0, num_points=1000)
        self.precompute_f_transitions(z_min=0.0, z_max=10.0, num_points=1000)

    def compute_Delta_c(self, z):
        """
        Parameters:
        - z: Redshift
        
        Returns:
        - Delta_c(z): Critical overdensity [dimensionless]
        """
        #Omega_M(z)
        Omega_M_z = self.omega_m * (1.+z) ** 3.0 / (self.omega_m * (1. + z)**3 + self.omega_l)
        x = Omega_M_z - 1
        Delta_c = 18 * np.pi**2 + 82 * x - 39 * x**2

        return Delta_c

    def compute_f_transitions(self, z):
        """
        Compute the transition frequencies f_merg, f_ring, sigma, and f_cut for given m1, m2 and z.
        
        Parameters:
        - z: Redshift
        
        Returns:
        - f_merg, f_ring, sigma, f_cut: Transition frequencies [Hz]
        """
        # Extract polynomial coefficients
        a_merg, b_merg, c_merg = self.polynomial_coeffs['f_merg_coeff']
        a_ring, b_ring, c_ring = self.polynomial_coeffs['f_ring_coeff']
        a_sigma, b_sigma, c_sigma = self.polynomial_coeffs['sigma_coeff']
        a_cut, b_cut, c_cut = self.polynomial_coeffs['f_cut_coeff']
        
        # symmetric mass ratio eta
        M = self.m1 + self.m2
        eta = (self.m1 * self.m2) / M**2
        
        # total mass in seconds: M_sec = G * M * M_sun / c**3
        M_sec = self.G_Mpc_Msun_s * M / self.c_light**3 # [s]
        
        # Compute transition frequencies using polynomial expressions
        f_merg = (a_merg * eta**2 + b_merg * eta + c_merg) / (np.pi * (1 + z) * M_sec)
        f_ring = (a_ring * eta**2 + b_ring * eta + c_ring) / (np.pi * (1 + z) * M_sec)
        f_cut = (a_cut * eta**2 + b_cut * eta + c_cut) / (np.pi * (1 + z) * M_sec)
        sigma = (a_sigma * eta**2 + b_sigma * eta + c_sigma) / (np.pi * (1 + z) * M_sec)
    
        return f_merg, f_ring, sigma, f_cut

    def tilde_h(self, f, z):
        """
        Compute the gravitational wave strain amplitude in the Fourier space \tilde{h}(f) for a binary SMBH merger.
        
        Parameters:
        - f: Frequency array in Hz
        - z: Redshift of the merger
        
        Returns:
        - h_tilde: Gravitational waveform in strain units
        """
        # Total mass and symmetric mass ratio
        M = self.m1 + self.m2 # Total mass [Msun]
        eta = (self.m1 * self.m2) / M**2 # Symmetric mass ratio

        # transition frequencies
        f_merg, f_ring, sigma, f_cut = self.compute_f_transitions(z)
        
        # normalization constant C

        # Luminoisty distance in Mpc
        D_lum = (1 + z) * self.comoving_distance(z)
        M_sec = self.G_Mpc_Msun_s / self.c_light**3 # [s]
        C_waveform = ((1 + z) * M * M_sec)**(5/6) * f_merg**(-7/6) / (D_lum * np.pi**(2/3)) * np.sqrt(5 * eta / 24)
        
        # weight w
        w = (np.pi * sigma / 2) * (f_ring / f_merg)**(-2/3)
        
        # Lorentzian function L(f, f_ring, sigma)
        L_f = (1 / (2 * np.pi)) * (sigma / ((f - f_ring)**2 + (sigma**2) / 4))

        # Initialize h_tilde
        h_tilde = np.zeros_like(self.m1)

        # Inspiral: f < f_merg
        inspiral_mask = f < f_merg
        h_tilde[inspiral_mask] = C_waveform[inspiral_mask] * (f / f_merg[inspiral_mask])**(-7/6)
        
        # Merger: f_merg <= f < f_ring
        merger_mask = (f >= f_merg) & (f < f_ring)
        h_tilde[merger_mask] = C_waveform[merger_mask] * (f / f_merg[merger_mask])**(-2/3)
        
        # Ringdown: f_ring <= f < f_cut
        ringdown_mask = (f >= f_ring) & (f < f_cut)
        h_tilde[ringdown_mask] = C_waveform[ringdown_mask] * w[ringdown_mask] * L_f[ringdown_mask]
        
        # Cutoff: f >= f_cut
        h_tilde[f >= f_cut] = 0.0

        # if f < f_merg: # Inspiral
        #     h_tilde = C_waveform * (f / f_merg)**(-7/6)
        # elif (f >= f_merg) & (f < f_ring): # Merger
        #     h_tilde = C_waveform * (f / f_merg)**(-2/3)
        # elif (f >= f_ring) & (f < f_cut): # Ringdown
        #     h_tilde = C_waveform * w * L_f
        # else:   #  >= cutoff
        #     h_tilde = 0.0
    
        return h_tilde

    # -------- Uncomment in do not need interpolations as done below --------
    # def comoving_distance(self, z):
    #     """
    #     Parameters:
    #     - z: Redshift

    #     Returns:
    #     - D_C: Comoving distance [Mpc]
    #     """

    #     # comoving distance D_C(z)
    #     D_C, _ = quad(lambda zp: self.c_light / self.H_z_function(zp), 0, z)

    #     return D_C # [Mpc]

    # def compute_dV(self, z):
    #     """
    #     Parameters:
    #     - z: Redshift
    #     - params: Dictionary containing cosmological parameters

    #     Returns:
    #     - dV: Comoving volume element [Mpc^3]
    #     """

    #     # H(z)
    #     H_z_val = self.H_z_function(z)  # s^-1
        
    #     D_C = self.comoving_distance(z)
        
    #     # Comoving volume element per steradian
    #     dV = (4 * np.pi * self.c_light * D_C**2) / H_z_val
        
    #     return dV

    # def H_z_function(self, z):
    #     """
    #     Compute H(z) [s^-1].

    #     Parameters:
    #     - z: Redshift
      
    #     Returns:
    #     - H_z: Hubble parameter at redshift z [s^-1]
    #     """
    #     return self.H0 * np.sqrt(self.omega_m * (1 + z)**3 + self.omega_l)
    
    # def cosmic_time(self, z):
    #     """
    #     Calculate cosmic time [in years] at redshift z.
    #     """
    #     # Function E(z)
    #     def E(zp):
    #         return np.sqrt(self.omega_l +
    #         (1.0 - self.omega_l - self.omega_m) * (1.0 + zp)**2 +
    #         self.omega_m * (1.0 + zp)**3 )

    #     # Integrate to compute cosmic time
    #     integrand = lambda zp: 1 / ((1 + zp) * E(zp))
    #     integral, error = quad(integrand, z, np.inf) # # Integrate from z to a very high redshift ~ infinity
    #     t = integral / (self.H0 * constants.yr_s) 
    #     return t  # in yrs


    # ------------------------ Interpolations ------------------------
    def precompute_Delta_c(self, z_min=0.0, z_max=10.0, num_points=300):
        """
        Precompute Delta_c(z) on a grid and build an interpolator.
        """
        z_grid = np.linspace(z_min, z_max, num_points)
        Delta_c_values = np.array([self.compute_Delta_c(z) for z in z_grid])
        self.Delta_c_interp = interp1d(z_grid, Delta_c_values, kind='cubic',
                                    bounds_error=False, fill_value="extrapolate")
        # self.z_grid_for_Delta_c = z_grid

    def compute_Delta_c_interp(self, z):
        """
        Return Delta_c using the precomputed interpolator if available.
        """
        if hasattr(self, 'Delta_c_interp'):
            return self.Delta_c_interp(z)
        else:
            return self.compute_Delta_c(z)
        
    def precompute_f_transitions(self, z_min=0.0, z_max=10.0, num_points=300):
        """
        Precompute the transition frequencies f_merg, f_ring, sigma, and f_cut on a grid of redshifts.
        We assume that self.m1 and self.m2 are fixed 2D meshgrids.
        For simplicity, we extract the first row (unique values) for interpolation.
        """
        z_grid = np.linspace(z_min, z_max, num_points)
        # For each z, compute the transitions.
        f_merg_vals = np.zeros(num_points)
        f_ring_vals = np.zeros(num_points)
        sigma_vals = np.zeros(num_points)
        f_cut_vals = np.zeros(num_points)
        
        for i, z in enumerate(z_grid):
            f_merg, f_ring, sigma, f_cut = self.compute_f_transitions(z)
            # Assuming that for fixed z, these arrays are constant (or nearly so) over the mass grid,
            # we can take one representative value (say, the first element).
            f_merg_vals[i] = f_merg.flat[0]
            f_ring_vals[i] = f_ring.flat[0]
            sigma_vals[i]   = sigma.flat[0]
            f_cut_vals[i]   = f_cut.flat[0]
        
        # Build interpolators for each transition frequency:
        self.f_merg_interp = interp1d(z_grid, f_merg_vals, kind='cubic',
                                    bounds_error=False, fill_value="extrapolate")
        self.f_ring_interp = interp1d(z_grid, f_ring_vals, kind='cubic',
                                    bounds_error=False, fill_value="extrapolate")
        self.sigma_interp   = interp1d(z_grid, sigma_vals, kind='cubic',
                                    bounds_error=False, fill_value="extrapolate")
        self.f_cut_interp   = interp1d(z_grid, f_cut_vals, kind='cubic',
                                    bounds_error=False, fill_value="extrapolate")
        # self.z_grid_for_ftrans = z_grid

    def compute_f_transitions_interp(self, z):
        """
        Return the transition frequencies at redshift z using the precomputed interpolators.
        """
        f_merg = self.f_merg_interp(z)
        f_ring = self.f_ring_interp(z)
        sigma   = self.sigma_interp(z)
        f_cut   = self.f_cut_interp(z)
        return f_merg, f_ring, sigma, f_cut
    
    def precompute_comoving_distance(self, z_min=0.0, z_max=10.0, num_points=300):
        """
        Precompute D_C(z) on a grid and build an interpolator.
        """
        z_grid = np.linspace(z_min, z_max, num_points)
        D_C_values = np.zeros_like(z_grid)
        
        for i, z in enumerate(z_grid):
            D_C_values[i], _ = quad(lambda zp: self.c_light / self.H_z_function(zp), 0, z)
        
        self.z_grid = z_grid 
        self.comoving_distance_interp = interp1d(z_grid, D_C_values, kind='cubic',
                                             bounds_error=False, fill_value="extrapolate")
    
    def comoving_distance(self, z):
        """
        Return comoving distance using the precomputed interpolator if available.
        """
        if hasattr(self, 'comoving_distance_interp'):
            return self.comoving_distance_interp(z)
        else:
            # When interpolator is not defined.
            D_C, _ = quad(lambda zp: self.c_light / self.H_z_function(zp), 0, z)
            return D_C
        
    def precompute_dV(self, z_min=0.0, z_max=10.0, num_points=300):
        """
        Precompute dV(z) on the same grid as for D_C(z) and build an interpolator.
        """
        if not hasattr(self, 'z_grid'):
            self.precompute_comoving_distance(z_min, z_max, num_points)
        
        z_grid = self.z_grid
        dV_values = np.zeros_like(z_grid)
        
        for i, z in enumerate(z_grid):
            H_z_val = self.H_z_function(z)
            D_C = self.comoving_distance(z)
            dV_values[i] = (4 * np.pi * self.c_light * D_C**2) / H_z_val
        
        self.dV_interp = interp1d(z_grid, dV_values, kind='cubic',
                              bounds_error=False, fill_value="extrapolate")
    
    def compute_dV(self, z):
        """
        Return dV(z) using the precomputed interpolator if available.
        """
        if hasattr(self, 'dV_interp'):
            return self.dV_interp(z)
        else:
            H_z_val = self.H_z_function(z)
            D_C = self.comoving_distance(z)
            return (4 * np.pi * self.c_light * D_C**2) / H_z_val
        

    def precompute_H_z(self, z_min=0.0, z_max=10.0, num_points=300):
        z_grid = np.linspace(z_min, z_max, num_points)
        H_z_values = self.H0 * np.sqrt(self.omega_m * (1 + z_grid)**3 + self.omega_l)
        self.H_z_interp = interp1d(z_grid, H_z_values, kind='cubic',
                                bounds_error=False, fill_value="extrapolate")
        
    def H_z_function(self, z):
        if hasattr(self, 'H_z_interp'):
            return self.H_z_interp(z)
        else:
            return self.H0 * np.sqrt(self.omega_m * (1 + z)**3 + self.omega_l)
        
    def precompute_cosmic_time(self, z_min=0.0, z_max=10.0, num_points=300):
        """
        Precompute cosmic time t(z) on a grid and build an interpolator.
        """
        z_grid = np.linspace(z_min, z_max, num_points)
        t_values = np.zeros_like(z_grid)
        
        for i, z in enumerate(z_grid):
            def E(zp):
                return np.sqrt(self.omega_l +
                            (1.0 - self.omega_l - self.omega_m) * (1.0 + zp)**2 +
                            self.omega_m * (1.0 + zp)**3)
            integrand = lambda zp: 1 / ((1 + zp) * E(zp))
            integral, _ = quad(integrand, z, np.inf)
            t_values[i] = integral / (self.H0 * constants.yr_s)
        
        self.cosmic_time_interp = interp1d(z_grid, t_values, kind='cubic',
                                       bounds_error=False, fill_value="extrapolate")
    
    def cosmic_time(self, z):
        """
        Return cosmic time using the precomputed interpolator if available.
        """
        if hasattr(self, 'cosmic_time_interp'):
            return self.cosmic_time_interp(z)
        else:
            def E(zp):
                return np.sqrt(self.omega_l +
                            (1.0 - self.omega_l - self.omega_m) * (1.0 + zp)**2 +
                            self.omega_m * (1.0 + zp)**3)
            integrand = lambda zp: 1 / ((1 + zp) * E(zp))
            integral, _ = quad(integrand, z, np.inf)
            return integral / (self.H0 * constants.yr_s)
        
    # ------------------------ End of interpolations -------------------------
    
    def integrand_z(self, z, f):
        """
        Compute the integrand for Omega_GW(f).
        
        Parameters:
        - f: Frequency [Hz]
        - z: Redshift
        
        Returns:
        - integrand_value: Contribution to Omega_GW at frequency f
        """
        # # Initialize BHMergerRateDensity
        # self.bh_merger_rate_density = BHMergerRateDensity(z)

        # Find the corresponding t_target
        t_at_z = self.cosmic_time(z)  # in yr

        # d2R_BH / dm1 dm2
        R_BH = self.bh_merger_rate_density.compute_R_BH(self.m1, self.m2, t_at_z) # [yr^-1 Msun^-2 Mpc^-3]

        dV = self.compute_dV(z)  # [Mpc^3]
        
        # |tilde{h}(f)|^2
        h_tilde = self.tilde_h(f, z)  # [strain]

        # Additional factor
        factor = (np.pi * self.c_light**2 * (f)**3) / (4 * self.G_Mpc_Msun_s * (1 + z))  # [Hz^3 (km/s)^2 / (m^3 kg^-1 s^-2)]
        
        # Compute the integrand
        density = dV * (R_BH/constants.yr_s) * factor *np.abs(h_tilde)**2  # [yr^-1 Msun^-2 Mpc^-3 * ...]
        integral_m1 = np.trapezoid(density, x = self.mass1, axis=1)  # Integrate over m1 for each m2
        integrand = np.trapezoid(integral_m1, x = self.mass2)  # Integrate over m2
        # print(f"z = {z:.1f}, R_BH = {integrand :.1e} yr^-1 Gpc^-3")
                
        return integrand  # [appropriate units]
    

    def compute_Omega_GW_single(self, f, z_max):
        """
        Compute Omega_GW for a single frequency.
        
        Parameters:
        - f: Frequency [Hz]
        - z_max: Maximum redshift

        Returns:
        - Omega_GW at frequency f
        """
        def integrand(z):
            return self.integrand_z(z, f)
        
        # Perform integral over z 
        Omega_GW_f, _ = quad(integrand, 1e-2, z_max, limit=10)
        
        return Omega_GW_f
    
    def compute_Omega_GW(self, f_array, z_max):
        """
        Compute Omega_GW(f) over a range of frequencies.
        
        Parameters:
        - f_array: Array of frequencies [Hz]
        - z_max  :  Max redshift to which to compute 
        
        Returns:
        - Omega_GW: Array of Omega_GW values corresponding to f_array
        """

        # self.bh_merger_rate_density = BHMergerRateDensity(0, self.m1, self.m2)
        # Omega_GW = np.zeros_like(f_array)
        # # Loop over frequencies
        # for idx, f in enumerate(f_array): # Integrante over z, m1, m2 for every f
        #     # for vectorization
        #     # f_val = np.full(m1.size, f)
        #     def integrand(z, f):
        #         return self.integrand_z(z, f)
        #     # Perform the integral over z using quad
        #     Omega_GW[idx], _ = quad(integrand, 0, z_max, limit = 100, args = f)
        # return Omega_GW / self.rho_c 

        self.bh_merger_rate_density = BHMergerRateDensity(0, self.m1, self.m2)
        Omega_GW = Parallel(n_jobs=-1)(
            delayed(self.compute_Omega_GW_single)(f, z_max) for f in f_array
        )
        
        Omega_GW = np.array(Omega_GW) / self.rho_c  # Normalize by critical density 
        
        return Omega_GW