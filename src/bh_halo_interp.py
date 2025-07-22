import os
import pickle
import hashlib
from scipy.interpolate import PchipInterpolator
from joblib import Parallel, delayed
import glob

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from astropy.cosmology import LambdaCDM, z_at_value
import astropy.units as u

from src.halo_mass_history import HaloMassHistory
from src.halo_mass_function import HaloMassFunction
from src.cosmolopy import constants

class BHhaloInterp():
    """Class for calculating the interpolated BH–halo mass relation
    for a range of redshifts and seed masses, with finite‐difference slopes
    and on-disk caching."""
    _cache = {}
    _unpickling = False   

    def __new__(cls, *args, **kwargs):
        # reproduce caching key
        z_final_vals = np.array(kwargs.get("z_final_vals", np.linspace(1e-2, 6., 10)))
        m_seeds      = np.atleast_1d(kwargs.get("m_seeds", np.logspace(4, 5.7, 10)))
        key = (
            tuple(z_final_vals),
            tuple(m_seeds),
            kwargs.get("z_seed", 10),
            kwargs.get("t_Edd", 45e7),
            kwargs.get("lambda_0", 0.003),
            kwargs.get("beta", 2.5),
            kwargs.get("epsilon", 0.1),
            kwargs.get("halo_mass_min", 1e7),
            kwargs.get("halo_mass_max", 1.93e9),
            kwargs.get("num_points", 10000),
            kwargs.get("h", 0.674),
            kwargs.get("omega_m0", 0.315),
            kwargs.get("omega_l0", 0.685)
        )

        # deterministic filename via md5 of repr(key)
        key_hash = hashlib.md5(repr(key).encode()).hexdigest()
        filename = os.path.expanduser(f"~/.cache/bhhalointerp/bhhalointerp_{key_hash}.pkl")

        if cls._unpickling:
            return super().__new__(cls)
        
        # Return from in-memory cache
        if key in cls._cache:
            return cls._cache[key]

        # On-disk cache
        if os.path.exists(filename):
            # tell ourselves “don’t re-enter this branch”
            cls._unpickling = True
            try:
                with open(filename, "rb") as f:
                    inst = pickle.load(f)
            finally:
                cls._unpickling = False
            # mark it initialized and stash
            inst._initialized = True
            cls._cache[key]   = inst
            return inst

        # Fresh instance
        inst = super().__new__(cls)
        # stash for save step
        inst._pickle_path = filename
        inst._cache_key   = key
        inst._initialized = False
        cls._cache[key]   = inst
        return inst

    def __init__(self, 
                 z_final_vals=np.linspace(1e-2, 6., 10), 
                 m_seeds=np.logspace(4, 5.7, 10),
                 z_seed=10,
                 t_Edd=45e7,
                 lambda_0=0.003,
                 beta=2.5,
                 epsilon=0.1,
                 halo_mass_min=1e7,
                 halo_mass_max=1.93e9,
                 num_points=10000,
                 h=0.674,
                 omega_m0=0.315,
                 omega_l0=0.685):
        
        # skip re-init if loaded or already built
        if self._initialized:
            print("Using previous instance of BHhaloInterp. \n" \
            "Use BHhaloInterp.clear_cache() to recompute - will take 6hrs+.")
            return
        self._initialized = True

        # store parameters
        self.z_final_vals = np.array(z_final_vals)
        self.m_seeds      = np.atleast_1d(m_seeds)
        self.z_seed       = z_seed
        self.t_Edd        = t_Edd
        self.lambda_0     = lambda_0
        self.beta         = beta
        self.epsilon      = epsilon
        self.halo_mass_min= halo_mass_min
        self.halo_mass_max= halo_mass_max
        self.num_points   = num_points
        self.h            = h
        self.omega_m0     = omega_m0
        self.omega_l0     = omega_l0

        # cosmology and H0 in yr^-1
        self.cosmo = LambdaCDM(H0=h*100, Om0=omega_m0, Ode0=omega_l0)
        self.H0    = h*100/constants.Mpc_km * constants.yr_s

        # grid definitions
        self.progenitor_halos = np.logspace(
            np.log10(halo_mass_min),
            np.log10(halo_mass_max),
            num_points
        )
        num_s = len(self.m_seeds)
        num_z = len(self.z_final_vals)
        num_M = len(self.progenitor_halos)

        # prepare grids
        self.bh_mass_grid   = np.empty((num_s, num_z, num_M))
        self.halo_mass_grid = np.empty((   num_z,     num_M))

        self.hmf0 = HaloMassFunction(0)

        # # fill halo_mass_grid
        # for i, zf in enumerate(self.z_final_vals):
        #     for j, M in enumerate(self.progenitor_halos):
        #         self.halo_mass_grid[i, j] = (
        #             HaloMassHistory(M, self.z_seed, hmf0)
        #             .mass_at_z_from_zi(zf)
        #         )

        # # fill black‐hole masses
        # for s, m0 in enumerate(self.m_seeds):
        #     for i, zf in enumerate(self.z_final_vals):
        #         for j, M in enumerate(self.progenitor_halos):
        #             self.bh_mass_grid[s, i, j] = (
        #                 self.black_hole_mass_at_z(M, zf, m_seed=m0)
        #             )

        # Halo‐mass grid: parallelize over z slices
        def _compute_halo_row(zf):
            # returns a length‐num_M row of halo masses at redshift zf
            return [
                HaloMassHistory(M, self.z_seed, self.hmf0)
                    .mass_at_z_from_zi(zf)
                for M in self.progenitor_halos
            ]

        rows = Parallel(n_jobs=-1, verbose=5)(
            delayed(_compute_halo_row)(zf)
            for zf in self.z_final_vals
        )
        # rows has length num_z, each an array of length num_M
        self.halo_mass_grid = np.vstack(rows)     # shape (num_z, num_M)


        # 2) BH‐mass grid: parallelize over seed‐slices

        def _compute_bh_slice(s):
            # returns a (num_z, num_M) block for seed index s
            m0 = self.m_seeds[s]
            block = np.empty((len(self.z_final_vals), len(self.progenitor_halos)))
            for i, zf in enumerate(self.z_final_vals):
                for j, M in enumerate(self.progenitor_halos):
                    block[i, j] = self.black_hole_mass_at_z(M, zf, m_seed=m0)
            return s, block

        slices = Parallel(n_jobs=-1, verbose=5)(
            delayed(_compute_bh_slice)(s)
            for s in range(len(self.m_seeds))
        )

        # pack them back into 3D array
        for s, block in slices:
            self.bh_mass_grid[s, :, :] = block

        self.build_interpolators_parallel()

        # save to disk so next time __new__ will load it
        os.makedirs(os.path.dirname(self._pickle_path), exist_ok=True)
        with open(self._pickle_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Cached BHhaloInterp to {self._pickle_path}")

    def _build_one_interp(self, s, i, m_seed):
        """
        Helper to build both the BH→halo interp and the slope interp
        for a single (seed‐index, z‐index) cell.
        """
        bh = self.bh_mass_grid[s, i, :]
        hm = self.halo_mass_grid[   i, :]

        # log–log
        logbh = np.log10(bh)
        loghm = np.log10(hm)

        # ensure strictly increasing
        if not np.all(np.diff(logbh) > 0):
            idx = np.argsort(logbh)
            logbh, loghm = logbh[idx], loghm[idx]
            bh,   hm     = bh[idx],   hm[idx]

        # finite‐difference in log–log
        # dloghm_dlogbh = np.gradient(loghm, logbh)

        # build a PCHIP in log–log
        pchip = PchipInterpolator(logbh, loghm, extrapolate=True)

        # derivative d(log hm)/d(log bh)
        dloghm_dlogbh = pchip.derivative()(logbh)

        # mask plateau at the seed
        eps = 0.005  # 0.05%
        flat = np.abs(logbh - np.log10(m_seed)) < np.log10(1+eps)
        dloghm_dlogbh[flat] = 0

        # convert back to dM/dm
        slopes = (10**(loghm - logbh)) * dloghm_dlogbh

        # build the two interp1d’s
        bh2halo = interp1d(
            bh, hm,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        dMdm    = interp1d(
            bh, slopes,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        return s, i, bh2halo, dMdm

    def build_interpolators_parallel(self):
        num_s = len(self.m_seeds)
        num_z = len(self.z_final_vals)

        # Prepare output containers
        self.BHhalo_interps = [[None]*num_z for _ in range(num_s)]
        self.dM_dm_interps  = [[None]*num_z for _ in range(num_s)]

        # Create a list of (s,i,m_seed) tasks
        tasks = [
            (s, i, self.m_seeds[s])
            for s in range(num_s)
            for i in range(num_z)
        ]

        # Run in parallel
        results = Parallel(n_jobs=-1, verbose=5)(
            delayed(self._build_one_interp)(s, i, m_seed)
            for s, i, m_seed in tasks
        )

        # Stitch results back into 2D lists
        for s, i, bh2halo, dMdm in results:
            self.BHhalo_interps[s][i] = bh2halo
            self.dM_dm_interps[s][i]  = dMdm
            
    def M_of_m(self, bh_mass, z_final, m_seed):
        """Return halo mass for given BH mass(es), redshift, and seed mass."""
        idx_z = np.argmin(np.abs(self.z_final_vals - z_final))
        idx_s = np.argmin(np.abs(self.m_seeds      - m_seed))
        halo_m = self.BHhalo_interps[idx_s][idx_z](bh_mass)
        return np.minimum(halo_m, 2.5e15)

    def dM_dm(self, bh_mass, z_final, m_seed):
        """Return dM/dm for given BH mass(es), redshift, and seed mass."""
        idx_z = np.argmin(np.abs(self.z_final_vals - z_final))
        idx_s = np.argmin(np.abs(self.m_seeds      - m_seed))
        return self.dM_dm_interps[idx_s][idx_z](bh_mass)
    
    def black_hole_mass_at_z(self, M, z_fin, m_seed):
        """Compute final BH mass at z_fin for progenitor halo M and seed m_seed."""
        t_start = self.cosmic_time(self.z_seed)
        t_end   = self.cosmic_time(z_fin)
        hmf0    = HaloMassFunction(0)

        def integrand(t, M0, z0):
            z_cur = z_at_value(self.cosmo.age, (t/1e9) * u.Gyr)
            M_new = HaloMassHistory(M0, z0, hmf0).mass_at_z_from_zi(z_cur)
            lam   = self.lambda_M(z_cur, M_new, self.lambda_0, self.beta, m_seed)
            return lam * (1 - self.epsilon) / (self.epsilon * self.t_Edd)

        integral, _ = quad(integrand, t_start, t_end, args=(M, self.z_seed), epsabs=1e-6, epsrel=1e-6, limit=250)
        return m_seed * np.exp(integral)
    
    def lambda_M(self, z, M, lambda_0, beta, m_seed):
        """Calculate the Eddington factor lambda """

        # if the mass of the halo is lower than the minimum allowed halo mass the BH can't grow (actually it won't even form)
        if M <= self.halo_mass_min:
            return 0.0
    
        lambda_star_z = lambda_0 * (1 + z)**beta
        m_fixed = 1e4
        M_break = 5e11 * (1 + z)**(-1.5) * (m_seed/m_fixed)**0.5
        delta_low = z/(0.5*self.z_seed)
        delta_high = z/(15*self.z_seed)
        if M < M_break:
            exponent = delta_low
            # beta = 2
        else:
            exponent = delta_high
            # beta = 1.5
        return lambda_star_z * (M / M_break)**exponent
    
    def cosmic_time(self, z):
        """
        Compute the cosmic time (in years) at redshift z using a simple 
        integration (with cosmolopy constants).
        """
        self.omega_m0 = 0.315
        self.omega_l0 = 0.685

        def E(zp):
            return np.sqrt(self.omega_l0 +
                        (1.0 - self.omega_l0 - self.omega_m0) * (1 + zp)**2 +
                        self.omega_m0 * (1 + zp)**3)
        
        integrand = lambda zp: 1 / ((1 + zp) * E(zp))
        integral, error = quad(integrand, z, np.inf)
        t = integral / self.H0
        return t
    
 
    @classmethod
    def clear_cache(cls):
        """Delete every bhhalointerp_*.pkl in the cache dir and clear the in-memory cache."""
        cache_dir = os.path.expanduser("~/.cache/bhhalointerp")
        pattern   = os.path.join(cache_dir, "bhhalointerp_*.pkl")

        # remove every matching file on disk
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass

        # reset in-memory cache
        cls._cache.clear()
        cls._unpickling = False
