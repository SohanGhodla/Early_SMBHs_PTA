from scipy.stats import rv_histogram
from src.cosmolopy import constants
import la_forge.core as co
import numpy as np
import pandas as pd
from scipy.stats import chi2

class sensitivityCurves:
    def __init__(self):
        self.h = 0.674  # Hubble parameter

    def convert_h_to_omega(self, excess_delay, freq):
        H0_hz = self.h * constants.H100_s
        Tspan = 505861299.1401644
        
        # Take absolute value of excess_delay and Calculate h_c^2(f)
        hc_squared = 12 * np.pi**2 * freq**3  * (10**(excess_delay))**2 * Tspan 
        
        # Calculate Omega_GW(f)
        omega_gw = (2 * np.pi**2 / (3 * H0_hz**2)) * freq**2 * hc_squared
        return omega_gw
    
    def calViolinPlot(self):
        data = 'PTA_data/'
        # Fluctuation frequencies in 15yr dataset
        Tspan = 505861299.1401644
        freq_violin = np.arange(1,15)/Tspan

        # free-spec MCMC chain
        corepath = data + '30fCP_30fiRN_3A_freespec_chain.core'
        freespec = co.Core(corepath=corepath)  # open fil
        params = freespec.params  # access parameter names
        # crn free spec param names
        hd_rho_labels = [f'gw_hd_log10_rho_{ii}' for ii in range(30)]  
        # accessing crn free spec rhos
        hd_chain = freespec(hd_rho_labels)  # returns array of burn-in chain


        # plotting violin plots
        rho_bins = np.linspace(-9, -4, num=400) # rho bins
        drho = rho_bins[1]-rho_bins[0]

        # turning freespec into rv hist and sampling
        epsilon = 1e-250  # make histograms for each rho
        hd_histograms = np.array([np.histogram(hd_chain[:,ii], 
                                            rho_bins, density=True)[0]
                            for ii in range(hd_chain.shape[1])])

        bin_mid = (rho_bins[1:] + rho_bins[:-1]) / 2.0
        rvhist = [rv_histogram([hist, rho_bins]) for hist in hd_histograms]

        # draw samples from histograms for each rho
        rv_samples_hd = [rvh.rvs(size=100000) for rvh in rvhist]

        rv_samples_hd = np.array(rv_samples_hd) 
        freq_violin = np.array(freq_violin) 

         # convert
        omega_violins = np.array([self.convert_h_to_omega(rv_samples_hd[i, :], freq_violin[i]) 
                                for i in range(len(freq_violin))])
        return freq_violin, omega_violins


    def calc_15yr_PTAcurves(self):
        # --------- PTA 15 yr sensitivity curve --------
        H0 = constants.H100_s * self.h # H0 in s^-1 
        dir = './PTA_data/'
        df = pd.read_csv(dir + 'sensitivity_curves_NG15yr_fullPTA.txt')
        omega_NANOGrav_15yr = 2 * np.pi**2 / 3 / (H0**2) * df['# Frequencies [Hz]']**2 * df['h_c [strain]']**2
        freq_PTA = df['# Frequencies [Hz]']

        return freq_PTA, omega_NANOGrav_15yr
    
    def fetchData(self):
        freq_violin, omega_violins = self.calViolinPlot()
        freq_PTA, omega_NANOGrav_15yr = self.calc_15yr_PTAcurves()
        return freq_violin, omega_violins,  freq_PTA, omega_NANOGrav_15yr
    
    def compute_chi2(self, omega_model, freq_model):
        """
        Compute chi square of model predictions against the PTA's violin distributions,
        assuming the violins are ~ Gaussian.

        Input:
        omega_model: i.e. the predicted values of \Omega_GW
        freq_model: the associated frequencies
        """

        freq_obs, omega_violins, _, _ = self.fetchData()

        #interpolate model onto the violin freqs
        omega_interp = np.interp(freq_obs, freq_model, omega_model)

        # compute empirical mean & variance at each freq
        mu     = np.mean(omega_violins, axis=1)  
        sigma2 = np.var(omega_violins, axis=1, ddof=1)   # unbiased var

        chi2_val = np.sum((omega_interp - mu)**2 / sigma2)

        dof       = len(freq_obs)
        # reduced value
        chi2_red  = chi2_val / dof

        # p-value from the chi square CDF
        p_value = 1 - chi2.cdf(chi2_val, df=dof)

        return chi2_val, chi2_red, p_value
    

    
