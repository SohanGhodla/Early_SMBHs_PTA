# -*- coding: utf-8 -*-
"""This module calculates the halo mass functions and is mainly taken from yt.
Three classes:
        HaloMassFunction: Calculates dn/dm, and dn /dln sigma
        overdensities: Calculates tables of sigma(M)
        TransferFunction: Eisenstein and Hu transfer function module"""

import numpy as np
import math
import scipy.integrate as integ
import scipy.interpolate
# import cosmolopy.constants as constants

class HaloMassFunction:
    """Module for calculating halo mass functions. Largely stolen from yt.

    Note self.mass_function(sigma) will return the mass function as defined in eq. 4 of Jenkins 2001.
    Default mass function is Sheth-Tormen.
    Mass function, f(sigma, z) = M / rho_0 (dn / d ln (sigma^-1))
    where rho0(z) is the mean density of the universe at redshift z
    n(M,z) is the abundance of halos with mass less than M at redshift z.

    dndm: returns dn/dm, which is the number of halos per unit mass
    mass_function: returns dn / dln sigma which is what is commonly referred to as the mass function.
    Other functions are various choices of mass function
    """
    def __init__(self, redshift, omega_m=0.315, omega_b=0.0493, omega_l=0.685, hubble=0.674, ns=0.9649, sigma8=0.811,  mass_function=None, log_mass_lim=(2, 20)):
        #Sigma tables
        self.overden = Overdensities(redshift,omega_m, omega_b,omega_l,hubble, ns,sigma8,log_mass_lim=log_mass_lim)
        if mass_function == None:
            self.mass_function = self.tinker
        else:
            self.mass_function = mass_function.__get__(self)

        self.delta_c0 = 1.69  # critical density for turnaround (Press-Schechter)
        self.omega_m0 = omega_m
        self.redshift = redshift
        self.h = hubble

    def dndm(self, mass):
        """Returns the halo mass function dn/dM in units of h^4 M_sun^-1 Mpc^-3
        Requires mass in units of M_sun /h """
        # Mean matter density at redshift z in units of h^2 Msolar/Mpc^3
        #This is rho_c in units of h^-1 M_sun (Mpc/h)^-3
        rhom = 2.78e+11* self.overden.omega_matter0 #self.overden.omega_matter_of_z(self.overden.redshift)
        sigma = self.overden.sigmaof_M_z(mass)
        mass_func = self.mass_function(sigma)
        #We have dn / dM = - d ln sigma /dM rho_0/M f(sigma)
        sigma_plus = self.overden.sigmaof_M_z(1.001*mass)
        dlogsigma = (np.log(sigma_plus)-np.log(sigma))/(0.001*mass)
        dndM = - dlogsigma*mass_func/mass*rhom
        # dn_lndM = - dlogsigma*mass_func/mass*rhom * mass
        # return dn_lndM  / self.h**3 / np.log(10) # the later divison for making the log base 10
        return dndM / self.h**4 # Now returns in M_sun^-1 Mpc^-3

    def press_schechter(self, sigma):
        """Press-Schechter (This form from Jenkins et al. 2001, MNRAS 321, 372-384, eqtn. 5)"""
        nu = self.delta_c0 / sigma
        return np.sqrt(2.0/np.pi) * nu * np.exp(-0.5*nu*nu)

    def jenkins(self, sigma):
        """Jenkins et al. 2001, MNRAS 321, 372-384, eqtn. 9"""
        return 0.315 * np.exp( -1.0 * np.power( np.abs( np.log(1.0/sigma) + 0.61), 3.8 ) )

    def sheth_tormen(self, sigma):
        """Sheth-Tormen 1999, eqtn 10, using expression from Jenkins et al. 2001, eqtn. 7"""
        nu = self.delta_c0 / sigma
        A=0.3222
        a=0.707
        p=0.3
        return A*np.sqrt(2.0*a/math.pi)*(1.0+ np.power( 1.0/(nu*nu*a), p) )*  nu * np.exp(-0.5*a*nu*nu)

    def warren(self, sigma):
        """LANL fitting function - Warren et al. 2005, astro-ph/0506395, eqtn. 5 """
        A = 0.7234
        a = 1.625
        b = 0.2538
        c = 1.1982
        return A*( np.power(sigma, -1.0*a) + b)*np.exp(-1.0*c / sigma / sigma )

    def tinker(self, sigma):
        """Tinker et al. 2008, eqn 3, Delta=200 # Delta=300"""
        # A = 0.2 
        # a = 1.52 
        # b = 2.25 
        # c = 1.27 

        A = 0.186
        a = 1.47
        b = 2.57
        c = 1.19
        return A * ( np.power((sigma / b), -a) + 1) * np.exp(-1 * c / sigma / sigma)

    def watson_FOF(self, sigma):
        """Watson + 2012, FOF halos linking length = 0.2 * mean"""
        A = 0.282
        a = 2.163
        b = 1.406
        c = 1.210
        return A*( np.power(b / sigma, 1.0*a) + 1)*np.exp(-1.0*c / sigma / sigma )

class Overdensities:
    """Module for calculating the linear theory overdensities.
    Main result obtained from sigmaof_M_z"""
    def __init__(self,redshift, omega_m=0.315, omega_b=0.0493, omega_l=0.685, hubble=0.674, ns=0.9649, sigma8=0.811, num_sigma_bins=50, log_mass_lim=(2, 20)):
        #Calculate the transfer functions
        self.TF = TransferFunction(omega_m, omega_b, 0.0, 0,omega_l, hubble, redshift)
        self.omega_matter0 = omega_m
        self.omega_lambda0 = omega_l
        self.hubble0 = hubble
        self.num_sigma_bins = num_sigma_bins
        self.ns = ns
        self.redshift = redshift
        self.log_mass_max = log_mass_lim[1]
        self.log_mass_min = log_mass_lim[0]

        # output arrays
        # 1) Virial radius in units of Mpc/h
        self.Rarray = np.empty(self.num_sigma_bins,dtype='float64')
        # 2) mass (Msolar/h)
        self.logmassarray = np.empty(self.num_sigma_bins, dtype='float64')
        # 4) sigma(M, z=0, where mass is in Msun/h)
        self.sigmaarray = np.empty(self.num_sigma_bins, dtype='float64')

        #WMAP pivot scale
        self.kpivot = 2e-3
        #Calculate the normalisation of the power spectrum
        self.Norm = 1.
        self.Norm = sigma8*sigma8 / self.sigma_squared_of_R(8.0)
        #Calculate and fill the sigmaarrays
        self.sigmaM()
        #Set up interpolator
        self.intp=scipy.interpolate.InterpolatedUnivariateSpline(self.logmassarray,self.sigmaarray)

    def sigmaof_M_z(self, M):
        """
        /* Main accessor function to get sigma(M)
        M in M_sun/h*/
        """
        return self.Dofz(self.redshift)*self.intp(np.log10(M))

    def sigmaM(self):
        """
         Written by BWO, 2006 (updated 25 January 2007).
         Converted to Python by Stephen Skory December 2009.

         This routine takes in cosmological parameters and creates a file (array) with
         sigma(M) in it, which is necessary for various press-schechter type
         stuff.  In principle one can calculate it ahead of time, but it's far,
         far faster in the long run to calculate your sigma(M) ahead of time.

         Inputs: cosmology, user must set parameters

         Outputs: four columns of data containing the following information:

         1) log mass (Msolar)
         2) mass (Msolar/h)
         3) Radius (comoving Mpc/h)
         4) sigma (normalized) using Msun/h as the input

         The arrays output are used later.
        """

        rho0 = self.omega_matter0 * 2.78e+11 # in units of h^2 Msolar/Mpc^3

        # spacing in mass of our sigma calculation
        dm = (float(self.log_mass_max) - self.log_mass_min)/self.num_sigma_bins

        #Loop over the total number of sigma_bins the user has requested.
        # For each bin, calculate mass and equivalent radius, and call
        # sigma_squared_of_R to get the sigma(R) (equivalent to sigma(M)),
        # normalize by user-specified sigma_8, and then write out.
        for i in range(self.num_sigma_bins):

            # thislogmass is in units of Msolar/h
            thislogmass = self.log_mass_min +  i*dm

            # mass in units of h^-1 Msolar
            thismass = 10.0**thislogmass

            # radius is in units of h^-1 Mpc (comoving)
            thisradius = math.pow( 3.0*thismass / 4.0 / math.pi / rho0, 1.0/3.0 )

            R = thisradius # h^-1 Mpc (comoving)

            self.Rarray[i] = thisradius  # h^-1 Mpc (comoving)
            self.logmassarray[i] = thislogmass  # log(Msun/h)

            # get normalized sigma(R)
            self.sigmaarray[i] = np.sqrt(self.sigma_squared_of_R(R))
            # All done!
        return

    def sigma_squared_of_R(self, R):
        """
        /* calculates sigma^2(R).  This is the routine where the magic happens (or
           whatever it is that we do here).  Integrates the sigma_squared_integrand
           parameter from 0 to infinity.
           Note that R is in h^-1 Mpc (comoving)
        */
        """
        result = integ.quad(self.sigma_squared_integrand,0,np.inf, args=(R,),epsrel=1e-2)[0]

        sigmasquaredofR = result / (2.0 * math.pi**2)

        return sigmasquaredofR

    def sigma_squared_integrand(self,k,R):
        """
        integrand for integral to get sigma^2(R).
        Parameters: Rcom is R in comoving Mpc/h
        """
        f = k**2*self.PofK(k)*self.WofK(R*k)**2.0
        return f

    def PofK(self,k):
        """
        /* returns power spectrum as a function of wavenumber k */
        """
        return self.Norm*k*(k/self.kpivot)**(self.ns-1) * self.TF.TFmdm_onek_hmpc(k)**2

    def TofK(self,k):
        """
        /* returns transfer function as a function of wavenumber k. */
        """
        thisTofK = self.TF.TFmdm_onek_hmpc(k)

        return thisTofK

    def WofK(self,x):
        """
        returns W(k*R), which is the fourier transform of the top-hat function.
        """
        #Series expansion actually good until kr~1
        if x < 1e-2:
            return 1./3. - x**2/30. +x**4/840.
        else:
            return 3.0 * (np.sin(x) - x*np.cos(x)) / x**3

    def Dofz(self, redshift):
        """
        /* Growth function */
        """

        thisDofz = self.gofz(redshift) / self.gofz(0.0) / (1.0+redshift)

        return thisDofz


    def gofz(self, redshift):
        """
        /* g(z) - I don't think this has any other name*/
        """

        thisgofz = 2.5 * self.omega_matter_of_z(redshift) / \
        ( math.pow( self.omega_matter_of_z(redshift), 4.0/7.0 ) - \
          self.omega_lambda_of_z(redshift) + \
          ( (1.0 + self.omega_matter_of_z(redshift) / 2.0) * \
          (1.0 + self.omega_lambda_of_z(redshift) / 70.0) ))

        return thisgofz


    def omega_matter_of_z(self,redshift):
        """
        /* Omega matter as a function of redshift */
        """

        thisomofz = self.omega_matter0 * math.pow( 1.0+redshift, 3.0) / \
            math.pow( self.Eofz(redshift), 2.0 )

        return thisomofz

    def omega_lambda_of_z(self,redshift):
        """
        /* Omega lambda as a function of redshift */
        """

        thisolofz = self.omega_lambda0 / math.pow( self.Eofz(redshift), 2.0 )

        return thisolofz

    def Eofz(self, redshift):
        """
        /* E(z) - I don't think this has any other name */
        """
        thiseofz = math.sqrt( self.omega_lambda0 \
            + (1.0 - self.omega_lambda0 - self.omega_matter0)*math.pow( 1.0+redshift, 2.0) \
            + self.omega_matter0 * math.pow( 1.0+redshift, 3.0)  )

        return thiseofz


class TransferFunction(object):
    """
    Fitting Formulae for CDM + Baryon + Massive Neutrino (MDM) cosmologies.
    Daniel J. Eisenstein & Wayne Hu, Institute for Advanced Study

    There are two primary routines here, one to set the cosmology, the
    other to construct the transfer function for a single wavenumber k.
    You should call the former once (per cosmology) and the latter as
    many times as you want.

        TFmdm_set_cosm() -- User passes all the cosmological parameters as
    	arguments the routine sets up all of the scalar quantites needed
    	computation of the fitting formula.  The input parameters are:
    	1) omega_matter -- Density of CDM, baryons, and massive neutrinos,
    				in units of the critical density.
    	2) omega_baryon -- Density of baryons, in units of critical.
    	3) omega_hdm    -- Density of massive neutrinos, in units of critical
    	4) degen_hdm    -- (Int) Number of degenerate massive neutrino species
    	5) omega_lambda -- Cosmological constant
    	6) hubble       -- Hubble constant, in units of 100 km/s/Mpc
    	7) redshift     -- The redshift at which to evaluate

       TFmdm_onek_mpc() -- User passes a single wavenumber, in units of Mpc^-1.
    	Routine returns the transfer function from the Eisenstein & Hu
    	fitting formula, based on the cosmology currently held in the
    	internal variables.  The routine returns T_cb (the CDM+Baryon
    	density-weighted transfer function).

       TFmdm_onek_mpc_nu() -- As above but returning the CDM+Baryon+neutrino
        transfer function

       TFmdm_master_onek_mpc_nu() -- As above but returning the master
        transfer function

    We also supply TFmdm_onek_hmpc(), which is identical to the previous
    	routine, but takes the wavenumber in units of h Mpc^-1.

    We hold the internal scalar quantities in global variables, so that
    the user may access them in an external program, via "extern" declarations.

    Please note that all internal length scales are in Mpc, not h^-1 Mpc!
    """

    def __init__(self, omega_matter, omega_baryon, omega_hdm,
	    degen_hdm, omega_lambda, hubble, redshift):
        """
        /* This routine takes cosmological parameters and a redshift and sets up
        all the internal scalar quantities needed to compute the transfer function. */
        /* INPUT: omega_matter -- Density of CDM, baryons, and massive neutrinos,
                        in units of the critical density. */
        /* 	  omega_baryon -- Density of baryons, in units of critical. */
        /* 	  omega_hdm    -- Density of massive neutrinos, in units of critical */
        /* 	  degen_hdm    -- (Int) Number of degenerate massive neutrino species */
        /*        omega_lambda -- Cosmological constant */
        /* 	  hubble       -- Hubble constant, in units of 100 km/s/Mpc */
        /*        redshift     -- The redshift at which to evaluate */
        /* OUTPUT: Returns 0 if all is well, 1 if a warning was issued.  Otherwise,
            sets many global variables for use in TFmdm_onek_mpc() */
        """
        self.qwarn = 0
        self.theta_cmb = 2.728/2.7 # Assuming T_cmb = 2.728 K

        # Look for strange input
        if (omega_baryon<0.0):
            raise ValueError("TFmdm_set_cosm(): Negative omega_baryon set to trace amount.\n")
        if (omega_hdm<0.0):
            raise ValueError("TFmdm_set_cosm(): Negative omega_hdm set to trace amount.\n")
        if (hubble<=0.0):
            raise ValueError("TFmdm_set_cosm(): Negative Hubble constant illegal.\n")
        elif (hubble>2.0):
            raise ValueError("TFmdm_set_cosm(): Hubble constant should be in units of 100 km/s/Mpc.\n")
        if (redshift<=-1.0):
            raise ValueError("TFmdm_set_cosm(): Redshift < -1 is illegal.\n")
        elif (redshift>99.0):
            raise ValueError("TFmdm_set_cosm(): Large redshift entered.  TF may be inaccurate.\n")

        if (degen_hdm<1):
            degen_hdm=1
        self.num_degen_hdm = degen_hdm
        # Have to save this for TFmdm_onek_mpc()
        # This routine would crash if baryons or neutrinos were zero,
        # so don't allow that.
        if (omega_baryon<=0):
            omega_baryon=1e-5
        if (omega_hdm<=0):
            omega_hdm=1e-5

        self.omega_curv = 1.0-omega_matter-omega_lambda
        self.omhh = omega_matter*hubble**2
        self.obhh = omega_baryon*hubble**2
        self.onhh = omega_hdm*hubble**2
        self.f_baryon = omega_baryon/omega_matter
        self.f_hdm = omega_hdm/omega_matter
        self.f_cdm = 1.0-self.f_baryon-self.f_hdm
        self.f_cb = self.f_cdm+self.f_baryon
        self.f_bnu = self.f_baryon+self.f_hdm

        # Compute the equality scale.
        self.z_equality = 25000.0*self.omhh/(self.theta_cmb)**4 # Actually 1+z_eq
        self.k_equality = 0.0746*self.omhh/(self.theta_cmb)**2

        # Compute the drag epoch and sound horizon
        z_drag_b1 = 0.313*math.pow(self.omhh,-0.419)*(1+0.607*math.pow(self.omhh,0.674))
        z_drag_b2 = 0.238*math.pow(self.omhh,0.223)
        self.z_drag = 1291*math.pow(self.omhh,0.251)/(1.0+0.659*math.pow(self.omhh,0.828))* \
            (1.0+z_drag_b1*math.pow(self.obhh,z_drag_b2))
        self.y_drag = self.z_equality/(1.0+self.z_drag)

        self.sound_horizon_fit = 44.5*math.log(9.83/self.omhh)/math.sqrt(1.0+10.0*math.pow(self.obhh,0.75))

        # Set up for the free-streaming & infall growth function
        self.p_c = 0.25*(5.0-math.sqrt(1+24.0*self.f_cdm))
        self.p_cb = 0.25*(5.0-math.sqrt(1+24.0*self.f_cb))

        omega_denom = omega_lambda+(1.0+redshift)**2*(self.omega_curv+\
                omega_matter*(1.0+redshift))
        self.omega_lambda_z = omega_lambda/omega_denom
        self.omega_matter_z = omega_matter*(1.0+redshift)**2*(1.0+redshift)/omega_denom
        self.growth_k0 = self.z_equality/(1.0+redshift)*2.5*self.omega_matter_z/ \
            (math.pow(self.omega_matter_z,4.0/7.0)-self.omega_lambda_z+ \
            (1.0+self.omega_matter_z/2.0)*(1.0+self.omega_lambda_z/70.0))
        self.growth_to_z0 = self.z_equality*2.5*omega_matter/(math.pow(omega_matter,4.0/7.0) \
            -omega_lambda + (1.0+omega_matter/2.0)*(1.0+omega_lambda/70.0))
        self.growth_to_z0 = self.growth_k0/self.growth_to_z0

        # Compute small-scale suppression
        self.alpha_nu = self.f_cdm/self.f_cb*(5.0-2.*(self.p_c+self.p_cb))/(5.-4.*self.p_cb)* \
        math.pow(1+self.y_drag,self.p_cb-self.p_c)* \
        (1+self.f_bnu*(-0.553+0.126*self.f_bnu*self.f_bnu))/ \
        (1-0.193*math.sqrt(self.f_hdm*self.num_degen_hdm)+0.169*self.f_hdm*math.pow(self.num_degen_hdm,0.2))* \
        (1+(self.p_c-self.p_cb)/2*(1+1/(3.-4.*self.p_c)/(7.-4.*self.p_cb))/(1+self.y_drag))
        self.alpha_gamma = math.sqrt(self.alpha_nu)
        self.beta_c = 1/(1-0.949*self.f_bnu)
        # Done setting scalar variables
        self.hhubble = hubble # Need to pass Hubble constant to TFmdm_onek_hmpc()

    def TFmdm_onek_mpc(self,  kk):
        """
        /* Given a wavenumber in Mpc^-1, return the transfer function for the
        cosmology held in the global variables. */
        /* Input: kk -- Wavenumber in Mpc^-1 */
        /* Output:
            growth_cb -- the transfer function for density-weighted
                    CDM + Baryon perturbations.
        """

        qq = kk/self.omhh*(self.theta_cmb)**2

        # Compute the scale-dependent growth functions
        y_freestream = 17.2*self.f_hdm*(1+0.488*math.pow(self.f_hdm,-7.0/6.0))* \
            (self.num_degen_hdm*qq/self.f_hdm)**2
        temp1 = math.pow(self.growth_k0, 1.0-self.p_cb)
        temp2 = np.power(self.growth_k0/(1+y_freestream),0.7)
        growth_cb = np.power(1.0+temp2, self.p_cb/0.7)*temp1

        tf_master=self.TFmdm_master_onek_mpc(kk)

        # Now compute the CDM+HDM+baryon transfer functions
        tf_cb = tf_master*growth_cb/self.growth_k0
        return tf_cb

    def TFmdm_master_onek_mpc(self,  kk):
        """
        Given a wavenumber in Mpc^-1, return the master
        transfer function
        Input: kk -- Wavenumber in Mpc^-1
        Output:
            tf_cb -- the transfer function for density-weighted
                    CDM + Baryon perturbations.
        """

        qq = kk/self.omhh*(self.theta_cmb)**2

        # Compute the master function
        gamma_eff = self.omhh*(self.alpha_gamma+(1-self.alpha_gamma)/ \
            (1+(kk*self.sound_horizon_fit*0.43)**4))
        qq_eff = qq*self.omhh/gamma_eff

        tf_sup_L = np.log(2.71828+1.84*self.beta_c*self.alpha_gamma*qq_eff)
        tf_sup_C = 14.4+325/(1+60.5*np.power(qq_eff,1.11))
        tf_sup = tf_sup_L/(tf_sup_L+tf_sup_C*(qq_eff)**2)

        qq_nu = 3.92*qq*math.sqrt(self.num_degen_hdm/self.f_hdm)
        max_fs_correction = 1+1.2*math.pow(self.f_hdm,0.64)*math.pow(self.num_degen_hdm,0.3+0.6*self.f_hdm)/ \
            (np.power(qq_nu,-1.6)+np.power(qq_nu,0.8))
        tf_master = tf_sup*max_fs_correction

        return tf_master

    def TFmdm_onek_mpc_nu(self,  kk):
        """
        Given a wavenumber in Mpc^-1, return the transfer function for the
        cosmology held in the global variables.
        Input: kk -- Wavenumber in Mpc^-1
        Output:
            tf_cbnu -- the transfer function for density-weighted
                    CDM + Baryon + Massive Neutrino perturbations.
        """

        qq = kk/self.omhh*(self.theta_cmb)**2

        # Compute the scale-dependent growth functions
        y_freestream = 17.2*self.f_hdm*(1+0.488*math.pow(self.f_hdm,-7.0/6.0))* \
            (self.num_degen_hdm*qq/self.f_hdm)**2
        temp1 = math.pow(self.growth_k0, 1.0-self.p_cb)
        temp2 = np.power(self.growth_k0/(1+y_freestream),0.7)
        growth_cbnu = np.power(np.power(self.f_cb,0.7/self.p_cb)+temp2, self.p_cb/0.7)*temp1

        tf_master=self.TFmdm_master_onek_mpc(kk)

        # Now compute the CDM+HDM+baryon transfer functions
        tf_cbnu = tf_master*growth_cbnu/self.growth_k0
        return tf_cbnu

    def TFmdm_onek_hmpc(self, kk):
        """
        /* Given a wavenumber in h Mpc^-1, return the transfer function for the
        cosmology held in the global variables. */
        /* Input: kk -- Wavenumber in h Mpc^-1 */
        /* Output: The following are set as global variables:
            growth_cb -- the transfer function for density-weighted
                    CDM + Baryon perturbations.
            growth_cbnu -- the transfer function for density-weighted
                    CDM + Baryon + Massive Neutrino perturbations. */
        /* The function returns growth_cb */
        """
        return self.TFmdm_onek_mpc(kk*self.hhubble)

