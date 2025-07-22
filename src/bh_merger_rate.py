import os
import pickle
import hashlib
import numpy as np
from scipy.integrate import quad
from scipy.integrate import quad_vec
# from scipy.interpolate import interp1d
from scipy.stats import lognorm
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
from astropy.cosmology import LambdaCDM, z_at_value
import astropy.units as u
from scipy.optimize import brentq
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
import glob
from scipy.stats import norm


from src.cosmolopy import constants
from src.halo_mass_function import HaloMassFunction
from src.halo_merger_rate import HaloMergerRateDensity
# from src.BH_halo_mass_interp import BHHaloMappingInterpolator
from src.halo_mass_history import HaloMassHistory
from src.bh_halo_interp import BHhaloInterp


class BHMergerRateDensity:
    """
    Class to calculate the comoving black hole merger rate density R_BH using the Extended Press-Schechter (EPS) formalism.
    The merger rate is computed as d² R_BH / (d m1 d m2) with units of Mpc^{-3} yr^{-1} Msun^{-2}.
    """
    _cache = {}
    _unpickling = False

    def __new__(cls, *args, **kwargs):
        # If in the middle of unpickling, don’t touch args at all
        # Always allow a “no-arg” __new__ for pickling
        if cls._unpickling or len(args) < 3:
            # pickle (or joblib) is calling with no args ? -- just give back an empty instance
            return super().__new__(cls)

        z_merger, m1, m2 = args[:3]
        t_i = kwargs.get("t_i", 6e8)

        # build a key tuple
        key = (
            m1.shape,              
            float(m1[0, 0]),           # first BH mass
            float(m1[0, -1]),          # last BH mass
            m2.shape,              
            float(m2[0, 0]),           # first BH mass
            float(m2[-1, 0]),          # last BH mass
            t_i,
        )

        # hash the key
        key_hash = hashlib.md5(repr(key).encode()).hexdigest()
        cache_path = os.path.expanduser(f"~/.cache/bhmerger/bhmerger_{key_hash}.pkl")

        # In‐memory cache
        if cache_path in cls._cache:
            inst = cls._cache[cache_path]
            inst._initialized = True  
            return inst


        # On‐disk cache
        if os.path.exists(cache_path):
            cls._unpickling = True
            # print({key_hash})
            try:
                with open(cache_path, "rb") as f:
                    inst = pickle.load(f)
            finally:
                cls._unpickling = False
            inst._initialized = True   
            cls._cache[cache_path] = inst
            return inst

        # Fresh instance
        inst = super().__new__(cls)
        inst._cache_path        = cache_path
        inst._initialized = False
        cls._cache[cache_path]  = inst
        return inst
    
    def __init__(self, z_merger, m1, m2, t_i = 6e8):
        """
        Parameters:
        - z_merger: Redshift at which BHMergerRateDensity is evaluated
        - t_i: Initial time for integration (yr), default set to 1 Myr.
        """
        if self._initialized:
            print("Using previous instance of BHMergerRateDensity; skipping precompute. \n"
                "Use BHMergerRateDensity.clear_cache() to recompute - will take 10–20 mins.")
            return
        self._initialized = True

        self.z_merger = z_merger
        self.t_i = t_i  # Initial time in yr
        self.h = 0.674  # Hubble parameter 
        self.H0 = self.h * 100 / constants.Mpc_km * constants.yr_s # Hubble constant yr^-1
        self.omega_m0 = 0.315
        self.omega_l0 = 0.685

        self.d_iso = 0.1  # d_iso in Mpc 
        self.hmf_0 = HaloMassFunction(0)  # initialized object at redhift "zero"
        # Set up an astropy cosmology instance.
        self.cosmo = LambdaCDM(H0=self.h * 100, Om0=self.omega_m0, Ode0=self.omega_l0)

        # Instantiate the extended mapping interpolator.
        self.BHhalomapping = BHhaloInterp(z_final_vals=np.linspace(1e-2, 6, 10))

        # Precompute the Rh values (saves time)
        if not hasattr(self, 'Rh_interpolator'):
            self.precompute_Rh_interpolator(m1, m2)

        # Precompute the p_merg interpolation 
        if not hasattr(self, '_p_merg_dist'): 
            self.p_merg() 

        self.precompute_dndM_interpolator(m1)

        # finally, save to disk for next time
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Cached BHMergerRateDensity to {self._cache_path}")

    def precompute_dndM_interpolator(self, m):
        seeds  = self.BHhalomapping.m_seeds   # array of seed masses
        z_grid = np.logspace(-2.2, 1.0, 300)

        self.dndM_seeds         = seeds
        self.dndM_interpolators = []

        # loop over each seed mass
        for m_seed in seeds:
            # parallel compute dndM(z) for this seed
            dndM_list = Parallel(n_jobs=-1, backend="threading")(
                delayed(self.compute_dndM)(z, m, m_seed)
                for z in z_grid
            )
            dndM_arr      = np.array(dndM_list)               # shape (n_z, N_mass)
            dndM_flat     = dndM_arr.reshape(len(z_grid), -1)      # flatten the mass dims
            interp   = RegularGridInterpolator(
                          (z_grid,),
                          dndM_flat,
                          bounds_error=False,
                          fill_value=0.0
                      )
            self.dndM_interpolators.append(interp)

    def compute_dndM(self, z_prime, m, m_seed):
        hmf_z = HaloMassFunction(z_prime)
        M1    = self.M_of_m(m, z_prime, m_seed)

        threshold = self.M_of_m(m_seed * 2., z_prime, m_seed) 
        M1[M1 <= threshold] = threshold
        dn_dm = hmf_z.dndm(M1 / self.h)

        return dn_dm

    def p_occ(self, m, z_prime,
            m_seed, n_BH, mu, sigma, logM_min, zmax=10):
        """
        Occupation fraction at z_prime for BH masses m using threshold CDF approach.
        Parameters:
        -----------
        m           : array_like  BH masses in Msun
        z_prime     : float       redshift
        m_seed      : float       BH‐seed mass for the BH→halo mapping
        n_BH        : float       total seed‐halo comoving density [Mpc⁻³]
        mu          : float       ⟨M⟩ at z=zmax in log10
        sigma       : float       scatter of log10 M (dex)
        logM_min    : float       lower truncation of log10 M (dex)
        zmax  
        """
        # Map BH mass -> host‐halo mass
        M1 = self.M_of_m(m, z_prime, m_seed)
        threshold = self.M_of_m(m_seed * 2., z_prime, m_seed) 
        M1[M1 <= threshold] = threshold

        # Evolve the z=zmax log10-mean & lower bound to z_prime
        M0_mu  = 10**mu
        M0_min = 10**logM_min
        mu_z   = HaloMassHistory(M0_mu,  zmax, self.hmf_0).mass_at_z_from_zi(z_prime)
        Mmin_z = HaloMassHistory(M0_min, zmax, self.hmf_0).mass_at_z_from_zi(z_prime)

        # Fetch the halo mass function at z_prime for these M1
        idx       = np.argmin(np.abs(self.dndM_seeds - m_seed))
        dndM_f    = self.dndM_interpolators[idx]([z_prime])[0]
        phi       = dndM_f.reshape(M1.shape)

        # Compute the truncated-threshold CDF in log10 M
        logmu_z   = np.log10(mu_z)
        logmin_z  = np.log10(Mmin_z)
        delta     = (np.log10(M1) - logmu_z) / sigma
        delta_min = (logmin_z - logmu_z) / sigma

        cdf       = norm.cdf(delta)                   # Φ((logM - logmu)/σ)
        cdf_min   = norm.cdf(delta_min)               # Φ((logMmin - logmu)/σ)
        p_cdf     = (cdf - cdf_min) / (1.0 - cdf_min)  # truncated CDF
        p_cdf[M1 < Mmin_z] = 0.0                       # zero below absolute min

        # Solve for p_max by matching n_BH = ∫ p_cdf(M) φ(M) dM
        #  approximate integral with trapezoidal rule over unique M1
        M_list = M1[0]
        phi_list = phi[0]
        p_list   = p_cdf[0]
        integral   = np.trapezoid(p_list * phi_list, x = M_list)
        p_max      = n_BH / integral

        # Final occupation probability
        p_occ_val = np.clip(p_max * p_cdf, 0.0, 1.0)
        return p_occ_val

    # -------- Old one ----------
    # def p_occ(self, m, z_prime, M_seed, m_seed):
    #     """
    #     Occupation fraction: probability that a halo hosting a BH of mass m at z_prime
    #     is seeded, taking into account feedback.

    #     Parameters:
    #     -----------
    #     m        : array_like
    #             BH masses [Msun]
    #     z_prime  : float
    #             Redshift
    #     M_seed   : float
    #             Halo‐seed mass for the occupation threshold
    #     m_seed   : float
    #             BH‐seed mass for the BH→halo mapping
    #     """
        
    #     M1 = self.M_of_m(m, z_prime, m_seed)
    #     threshold = self.M_of_m(m_seed * 2., z_prime, m_seed) 
    #     M1[M1 <= threshold] = threshold

    #     # Compute the minimum halo mass for seeding
    #     z_max = 10
    #     mass_history_model = HaloMassHistory(M_seed, z_max, self.hmf_0)
    #     M_min = mass_history_model.mass_at_z_from_zi(z_prime)

    #     # Look up the right dndM‐interpolator for m_seed
    #     idx_s = np.argmin(np.abs(self.dndM_seeds - m_seed))
    #     # evaluate at z_prime (returns a flattened array)
    #     dndM_val_flat = self.dndM_interpolators[idx_s](np.array([z_prime]))[0]
    #     dndM_val = dndM_val_flat.reshape(M1.shape)   # back to the shape of M1

    #     # Integrate above M_min to get total number density
    #     mask0   = M1[0] >= M_min
        
    #     n_halo_z = np.trapezoid(dndM_val[0][mask0], x=M1[0][mask0])
    #     if n_halo_z <= 0:
    #         print(f'Max M1 = {np.max(M1[0]):.2e}, Min M1 = {np.min(M1[0]):.2e}')
    #         print(f"n_halo_z = {n_halo_z} at z={z_prime:.2f}, M_seed={M_seed:.2e}")
    #         # print(f'Masses large than M_min are {M1[0][mask0]} with M_min = {M_min:.2e}')

    #     # Compute the occupation fraction
    #     p_occ_value = np.zeros_like(M1)
    #     mask = M1 >= M_min
    #     p_occ_value[mask] = 2e-3 / n_halo_z   # tunable normalization
    #     p_occ_value = np.minimum(p_occ_value, 1.0)

    #     return p_occ_value
    
    def M_of_m(self, m, z, m_seed):
        """
        Return the host halo mass corresponding to a black hole mass at redshift z 
    
        Parameters:
          - m : black hole mass (or array) [M_sun].
          - z : redshift at which to compute the mapping.
          
        Returns:
          - Halo mass [M_sun].
        """
        if z >= 5. and m_seed >  1e5:
            return self.BHhalomapping.M_of_m(m, 5, m_seed)
        return self.BHhalomapping.M_of_m(m, z, m_seed)

    def dM_dm(self, m, z, m_seed):
        """
        Return the derivative dM/dm at a given BH mass for redshift z

        Parameters:
          - m : black hole mass (or array) [M_sun].
          - z : redshift.
          
        Returns:
          - dM/dm (unitless).
        """
        if z >= 5. and m_seed >  1e5:
            return self.BHhalomapping.dM_dm(m, 5, m_seed)
        return self.BHhalomapping.dM_dm(m, z, m_seed)

    # ----------- Used before MC implementation ---------
    # def p_merg(self, tau0=1.0, sigma=1/3):
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
    
    def p_merg(self, tau0=1.0, sigma=0.5):
        """
        Build and return a log-normal delay-time distribution for τ [in Gyr]:
          p(τ) = LogNorm(sigma, scale=exp(mu)),  mu = ln(tau0) - 0.5*sigma^2
        """
        # convert base-10 moments to natural-log moments
        ln10     = np.log(10)
        sigma_ln = sigma * ln10
        mu_ln    = np.log(tau0) - 0.5 * sigma_ln**2

        dist     = lognorm(s=sigma_ln, scale=np.exp(mu_ln))
        self._p_merg_dist = dist
        return dist
    
    
    def mass_seed(self, m_low, m_high, mean_log, sigma_dex = 1/3):
        """
        Parameters:
        m_low: lower range of mass
        m_high: upper range of mass
        mean: mean value of mass
        
        Returns:
        m_seed : float 
            A single seed mass in Msun, sampled from the truncated log-normal distribution.
        """
        mu    = mean_log
        sigma = sigma_dex

        # standardize the truncation limits
        a = (m_low  - mu) / sigma
        b = (m_high - mu) / sigma

        # draw one sample of X = log10(M)
        x_sample = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1)
        return 10**x_sample.item()

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

    def compute_Rh_at_z(self, z_prime, m1, m2, m_seed):
        # Build the halo–halo merger rate at z′ *for that BH‐seed mapping*
        hmf_z = HaloMassFunction(z_prime)
        hmr_z = HaloMergerRateDensity(hmf_z)
        # map each BH mass to its host‐halo mass using m_seed and z_prime
        M1 = self.M_of_m(m1, z_prime, m_seed)
        M2 = self.M_of_m(m2, z_prime, m_seed)

        threshold = self.M_of_m(m_seed * 2., z_prime, m_seed) 
        M1[M1 <= threshold] = threshold
        M2[M2 <= threshold] = threshold


        Rh = hmr_z.compute_Rh(M1/self.h, M2/self.h)
        return Rh

    # ------- Doesn't use threading - threading better for pickling ---------
    # def precompute_Rh_interpolator(self, m1, m2):
    #     n_z = 300
    #     z_prime_grid = np.logspace(-2.2, 1, n_z) 

    #     Rh_values_list = Parallel(n_jobs=-1)(
    #         delayed(self.compute_Rh_at_z)(z_prime, m1, m2) for z_prime in z_prime_grid
    #     )
    #     Rh_values = np.array(Rh_values_list)  # Shape: (n_z, N1, N2)

    #     # Reshape for interpolation:
    #     Rh_values_reshaped = Rh_values.reshape(n_z, -1)
        
    #     # Build the interpolator:
    #     self.Rh_interpolator = RegularGridInterpolator(
    #         (z_prime_grid,), Rh_values_reshaped, bounds_error=False, fill_value=0
    #     )

    def precompute_Rh_interpolator(self, m1, m2):
        seeds  = self.BHhalomapping.m_seeds   
        z_grid = np.logspace(-2.2, 1.0, 300)

        self.Rh_seeds = seeds
        self.Rh_interpolators = []

        for m_seed in seeds:
            # parallel over z′ to get Rh(z′)
            Rh_list = Parallel(n_jobs=-1, backend="threading")(
                delayed(self.compute_Rh_at_z)(z, m1, m2, m_seed)
                for z in z_grid
            )
            Rh_arr   = np.array(Rh_list)                  # shape (n_z, N1, N2)
            Rh_flat  = Rh_arr.reshape(len(z_grid), -1)    # flatten BH‐mass dims
            interp   = RegularGridInterpolator(
                        (z_grid,),
                        Rh_flat,
                        bounds_error=False,
                        fill_value=0.0
                    )
            self.Rh_interpolators.append(interp)

    # ------ Below uses quad_vec but not suitable if stochasticity in involved -------

    # def compute_R_BH(self, m1, m2, t):
    #     """
    #     Compute the comoving black hole merger rate density d²R_BH / (d m1 d m2).

    #     Parameters:
    #     - m1: Mass of the first black hole (Msun).
    #     - m2: Mass of the second black hole (Msun).
    #     - t: Current cosmic time (yr).

    #     Returns:
    #     - R_BH: The comoving black hole merger rate density in units of Mpc^{-3} yr^{-1} Msun^{-2}.
    #     """

    #     if t < self.t_i:
    #         raise ValueError("t_merger [i.e., t] is less than t_i")
        
    #     # Integrand as a function of t'
    #     def integrand(t_prime):
            
    #         z_prime = z_at_value(self.cosmo.age, (t_prime / 1e9) * u.Gyr).value
    #         # z_prime = self.redshift_from_time(t_prime)
    #         # print('z_prime =', z_prime, t_prime/1e9)

    #         m_seed_m1 = self.mass_seed(m_low = 4, m_high = 6, mean_log = 5)
    #         m_seed_m2 = self.mass_seed(m_low = 4, m_high = 6, mean_log = 5)

    #         # Compute R_h(t') = d^3 R_h / dM1 dM2 dt' at z_prime
    #         # find which precomputed interp to use
    #         idx_s = np.argmin(np.abs(self.Rh_seeds - m_seed_m1))
    #         # now fetch Rh from that one interpolator:
    #         Rh_flat = self.Rh_interpolators[idx_s](np.array([z_prime]))[0]
    #         # Rh_flat = self.Rh_interpolator(np.array([z_prime]))[0]  # Consider only first element as rest are identical (made N1 copies to assist vectorization)
    #         Rh = Rh_flat.reshape(m1.shape)

    #         # Compute occupation probabilities at z
    #         p_occ_m1 = self.p_occ(m1, z_prime, M_seed_1, m_seed_m1)
    #         # p_occ_m2 = p_occ_m1.T #self.p_occ(m2, z_prime)
    #         p_occ_m2 = self.p_occ(m1, z_prime, M_seed_2, m_seed_m2).T

    #         # Compute delay time tau = t - t'
    #         tau = t - t_prime  # in years
    #         tau_Gyr = tau / 1e9  # Convert years to Gyr

    #         # Compute merger probability with delay time
    #         p_merg = self.p_merg_interp(tau_Gyr)
            
    #         # Compute dM1/dm1 and dM2/dm2
    #         # hmf_z = HaloMassFunction(z_prime)
    #         dM1_dm1 = self.dM_dm(m1, z_prime, m_seed_m1) #dM/dm
    #         dM2_dm2 = self.dM_dm(m2, z_prime, m_seed_m2)
            
    #         if np.any(np.isnan(Rh)):
    #             print("Array contains NaN values in HaloMergerRate class")

    #         # Integrand: p_occ(m1, z') p_occ(m2, z') p_merg(m1, m2, tau) (dM1/dm1) (dM2/dm2) Rh(t')
    #         integrand_value = p_occ_m1 * p_occ_m2 * p_merg * dM1_dm1 * dM2_dm2 * Rh 

    #         return integrand_value

    #     # # Perform the integration from t_i to t 
    #     R_BH, err, info = quad_vec(integrand, self.t_i, t, limit = 50, full_output=True)
    #     # print("Function evaluations:", info.neval)
        # return R_BH  # Units:  Mpc^{-3} yr^{-1} Msun^{-2}
        
        
    def compute_R_BH(self, m1, m2, mu_val, t, tau0=1.0, sigma_tau=0.5, n_samples=5000):
        """
        MC-estimate of R_BH(m1,m2) via direct sampling from the full log-normal p_merg.
        Returns R_BH of shape m1.shape.
        """
        if t < self.t_i:
            raise ValueError("t_merger [i.e., t] is less than t_i")
        
        # Draw delay times [Gyr] -- log-normal distribution
        dist = self.p_merg(tau0, sigma_tau)
        tau_gyr  = dist.rvs(size=n_samples)  # draws on [0, inf) -- shape (n_samples,)

        # Convert to cosmic times [yr]
        t_primes = np.maximum(t - tau_gyr*1e9, self.t_i)  # shape (n_samples,)

        # Build the kernel K(tau_j) = f(t') / p_merg(tau_j):
        K = np.empty((n_samples,) + m1.shape)
        for j, t_p in enumerate(t_primes):
            # get redshift at t′
            z_p = z_at_value(self.cosmo.age, (t_p/1e9) * u.Gyr).value

            # random seed‐masses
            m_seed = self.mass_seed(m_low=4.0,  m_high=6.0,  mean_log=5.0)

            # Compute R_h(t') = d^3 R_h / dM1 dM2 dt' at z_prime
            # find which precomputed interp to use
            idx_s = np.argmin(np.abs(self.Rh_seeds - m_seed))
            # now fetch Rh from that one interpolator:
            Rh_flat = self.Rh_interpolators[idx_s](np.array([z_p]))[0]
            # Rh_flat = self.Rh_interpolator(np.array([z_prime]))[0] # Consider only first element as rest are identical (made N1 copies for vectorization)
            Rh = Rh_flat.reshape(m1.shape)

            p1 = self.p_occ(m1, z_p, m_seed, n_BH = 5e-3, mu = mu_val, sigma = 0.5, logM_min = 7)
            p2 = p1.T # don't use mass m2 as the first parameter (see p_occ's M1[0] term)
            d1 = self.dM_dm(m1, z_p, m_seed)
            d2 = d1.T # self.dM_dm(m2, z_p, m_seed)

            K[j] = p1 * p2 * d1 * d2 * Rh

        # Monte Carlo estimate and its standard error
        R_BH = K.mean(axis=0)
        err  = K.std(axis=0) / np.sqrt(n_samples)
        return R_BH  # Units:  Mpc^{-3} yr^{-1} Msun^{-2}
    
    @classmethod
    def clear_cache(cls):
        """
        Delete all on-disk cache files for BHMergerRateDensity
        and clear the in-memory cache.
        """
        cache_dir = os.path.expanduser("~/.cache/bhmerger")
        pattern   = os.path.join(cache_dir, "bhmerger_*.pkl")

        matches   = glob.glob(pattern)
        print("Using pattern:", pattern)
        print("Will delete these files:", matches)

        # remove every matching file on disk
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass

        # reset in-memory cache
        cls._cache.clear()
        cls._unpickling = False