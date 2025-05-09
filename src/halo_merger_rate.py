from src.halo_mass_function import HaloMassFunction #, Overdensities, TransferFunction
import numpy as np
from src.cosmolopy import constants


class HaloMergerRateDensity:
    """
    Class to calculate the halo merger rate using the Extended Press-Schechter (EPS) formalism.
    The merger rate is computed as d²Rh / (dM1 dM2) with units of Mpc^{-3} yr^{-1} Msun^{-2}.
    """
    def __init__(self, HaloMassFunction, dz=1e-3):
        """
        Initialize with an instance of HaloMassFunction.

        Parameters:
        - HaloMassFunction: An instance of HaloMassFunction initialized with desired cosmology and redshift.
        - dz: Step size for finite difference in redshift to compute derivatives.
        """
        self.hmf = HaloMassFunction
        self.h = self.hmf.h
        self.overden = self.hmf.overden
        self.delta_c0 = self.hmf.delta_c0
        self.dz = dz  # Step size for finite difference

    def compute_d2p_dt_dM2(self, M1, M2):
        """
        Compute the d2p_dt_dM2 for halos of mass M1 and M2 merging to form Mf = M1 + M2.

        Parameters:
        - M1: Mass of the smaller halo (Msun).
        - M2: Mass of the larger halo (Msun).

        Returns:
        - d2p_dt_dM2: The probabitiy of merger in units of yr^{-1} Msun^{-1}.
        """
        M_f = M1 + M2  # Final halo mass

        # Retrieve sigma values
        sigma_M_f = self.overden.sigmaof_M_z(M_f)
        sigma_M1 = self.overden.sigmaof_M_z(M1)

        # Numerical derivative d ln sigma / d ln M_f
        delta_logM = 1e-3
        logM_f = np.log(M_f)
        logM_f_plus = logM_f + delta_logM
        logM_f_minus = logM_f - delta_logM
        M_f_plus = np.exp(logM_f_plus)
        M_f_minus = np.exp(logM_f_minus)
        sigma_Mf_plus = self.overden.sigmaof_M_z(M_f_plus)
        sigma_Mf_minus = self.overden.sigmaof_M_z(M_f_minus)
        dln_sigma_dlnMf = np.abs(np.log(sigma_Mf_plus) - np.log(sigma_Mf_minus)) / (2 * delta_logM) #f'(x) = [f(x+h) - f(x - h)] / 2h as lim h -> 0.

        # Compute delta_crit(t)
        z = self.overden.redshift
        D_z = self.overden.Dofz(z)
        delta_crit = self.delta_c0 / D_z

        # Compute d delta_crit / dt
        dz = self.dz
        z_plus = z + dz
        z_minus = z - dz

        # Ensure z_minus remains positive
        if z_minus < 0:
            z_minus = 0.0

        D_z_plus = self.overden.Dofz(z_plus)
        D_z_minus = self.overden.Dofz(z_minus)

        if z_minus == 0.:
            dD_dz = (D_z_plus - D_z_minus) / dz #f'(x) = [f(x+h) - f(x)] / h as lim h -> 0.
        else:
            dD_dz = (D_z_plus - D_z_minus) / (2 * dz) #f'(x) = [f(x+h) - f(x - h)] / 2h as lim h -> 0.

        # Compute E(z) for H(z)
        E_z = self.overden.Eofz(z)
        H_z = self.h * 100.0 * E_z  # H(z) in km/s/Mpc

        # Convert H(z) from km/s/Mpc to yr^{-1}
        H_z_yr = H_z * (1.0 / constants.Mpc_km) * (constants.yr_s)  # yr^{-1}

        # Compute dD/dt
        dD_dt = -(1 + z) * H_z_yr * dD_dz

        # Compute d delta_crit / dt
        ddelta_crit_dt = -self.delta_c0 / (D_z ** 2) * dD_dt

        # Compute |d delta_crit / dt / delta_crit|
        abs_ddelta_crit_dt_over_delta_crit = np.abs(ddelta_crit_dt / delta_crit) 

        # Compute the term [1 - sigma^2(M_f)/sigma^2(M1)]^{-3/2}
        sigma_ratio_squared = (sigma_M_f ** 2) / (sigma_M1 ** 2)
        term = 1.0 - sigma_ratio_squared
        term_factor = term ** (-1.5)

        # Compute the exponential factor
        exponent = -0.5 * (delta_crit ** 2) * (1.0 / sigma_M_f ** 2 - 1.0 / sigma_M1 ** 2)
        exp_factor = np.exp(exponent)

        # Combine all terms to get the merger rate
        d2p_dt_dM2 = (
            (1.0 / M_f) *
            np.sqrt(2.0 / np.pi) *
            abs_ddelta_crit_dt_over_delta_crit *
            dln_sigma_dlnMf *
            term_factor *
            (delta_crit / sigma_M_f) *
            exp_factor
        )

        return d2p_dt_dM2
    
    def compute_Rh(self, M1, M2):
        """
        Compute the comoving halo merger rate density d²Rh / (dM1 dM2).

        Parameters:
        - M1: Mass of the larger halo (Msun).
        - M2: Mass of the smaller halo (Msun).

        Returns:
        - Rh: The comoving halo merger rate density in units of Mpc^{-3} yr^{-1} Msun^{-2}.
        """
        # Retrieve dn/dM1 and dn/dM2
        dn_dM1 = self.hmf.dndm(M1)  # Units: Msun^{-1} Mpc^{-3}
        dn_dM2 = self.hmf.dndm(M2)  # Units: Msun^{-1} Mpc^{-3}

        # # Compute Q(M1, M2, t)
        Q = self.compute_d2p_dt_dM2(M1, M2) / dn_dM2  # Units: yr^{-1} Msun^{-1}

        # Compute d²Rh / (dM1 dM2) = dn/dM1 * dn/dM2 * Q
        Rh = dn_dM1 * dn_dM2 * Q   # dn_dM1 * self.compute_d2p_dt_dM2(M1, M2)

        if np.any(np.isnan(Rh)):
            print("Array contains NaN values")
        # Apply the condition M2 > M1
        Rh[M2 >= M1] = 0.

        return Rh  # Units: yr^{-1} Msun^{-2} Mpc^{-3}