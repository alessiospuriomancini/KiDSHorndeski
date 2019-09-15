#!/usr/bin/env python
# encoding: UTF8
#
#######################################################################
# likelihood for the KiDS-450 WL power spectrum (quadratic estimator) #
#######################################################################
#
# Developed by A. Spurio Mancini and F. Koehlinger
#
#
#
# To be used with data from F. Koehlinger et al. 2017 (MNRAS;
# arXiv:1706.02892) which can be downloaded from:
#
# http://kids.strw.leidenuniv.nl/sciencedata.php
#
#######################################################################

from montepython.likelihood_class import Likelihood
import os
import numpy as np
import scipy.interpolate as itp
import scipy.integrate as integr
from scipy.linalg import cholesky, solve_triangular
import time

@np.vectorize
def _vecquad(function, low, high, **kwargs):

    integral, error = integr.quad(function, low, high, **kwargs)

    return integral, error

def vecquad(function, low, high, **kwargs):
    '''Integrate a function from low to high (vectorized).
    Vectorized convenience function.
    '''

    return _vecquad(function, low, high, **kwargs)

@np.vectorize
def _logquad(function, low, high, **kwargs):
    # Transform the function to log space.
    def func_dlnx (lnx):
        x = np.exp(lnx)
        return function(x) * x

    integral, error = integr.quad(func_dlnx, np.log(low), np.log(high), **kwargs)

    return integral, error

def logquad(function, low, high, **kwargs):
    """Integrate a function from low to high using a log transform (vectorized).

       The log transform is applied to the variable over which the
       integration is being performed.

    """
    return _logquad(function, low, high, **kwargs)



class MGfast(Likelihood):

    def __init__(self, path, data, command_line):
        # I should already take care of using only GRF mocks or data here (because of different folder-structures etc...)
        # or for now just write it for GRFs for tests and worry about it later...
        Likelihood.__init__(self, path, data, command_line)

        start_load = time.time()     # Starting time counter for loading data

        #self.need_cosmo_arguments(data, {'output':'mPk'})      # Parameters for Class

        ###### Alessio
        ######################################################################
        ###### Execution of operations for calculation of likelihood       ###
        ######################################################################


        # set type for interpolation (for quad and logquad):
        # 'linear', 'quadratic', or 'cubic'
        self.type_interp_nz = 'quadratic'

        # Alessio
        # Load the QE band powers and redshift distributions:
        self.__load_redshift_distributions()

        # FK: we need to define some numbers:
        # number of shear-shear correlations:
        self.nzcorrs_mm = self.nzbins_background * (self.nzbins_background + 1) / 2
        # number of galaxy-matter correlations:
        self.nzcorrs_gm = self.nzbins_foreground * self.nzbins_background

        # we set flags if some of the probes are not requested to skip the calculations:
        self.flag_no_MM = all(elem == 0 for elem in self.bands_EE_to_use_MM)
        self.flag_no_GM = all(elem == 0 for elem in self.bands_EE_to_use_GM)
        self.flag_no_GG = all(elem == 0 for elem in self.bands_EE_to_use_GG)

        # FK: get indices for masking
        self.mask_indices = self.__get_masking_indices()

        # FK: load the data-vector
        ell_bins, data_vec = self.__load_data_vector()
        self.nellsmax = len(self.bands_EE_to_use_MM)
        self.ell_bin_centers = ell_bins[:self.nellsmax]
        #self.ell_bin_centers = range(200,1501)
        #self.nellsmax = len(self.ell_bin_centers)
        self.band_powers = data_vec

        # FK: load the covariance matrix
        # also included (optional) masking
        covariance = self.__load_covariance_matrix()
        covariance = covariance[np.ix_(self.mask_indices, self.mask_indices)]
        # this is what we need for the chi2-calculation:
        self.cholesky_transform = cholesky(covariance, lower=True)

        self.path_save = '/home/alessio/spectralcdmlonger2/'
        # k_max is arbitrary at the moment, since cosmology module is not calculated yet...TODO
        """
        if self.mode == 'halofit':
            self.need_cosmo_arguments(data, {'z_max_pk': self.z_max, 'output': 'mPk', 'non linear': self.mode, 'P_k_max_h/Mpc': self.k_max_h_by_Mpc})
        else:
            self.need_cosmo_arguments(data, {'z_max_pk': self.z_max, 'output': 'mPk', 'P_k_max_h/Mpc': self.k_max_h_by_Mpc})
        """
        # set additional parameters for CLASS:
        self.need_cosmo_arguments(data, {'z_max_pk': self.z_max, 'output': 'mPk nCl', 'P_k_max_h/Mpc': self.k_max_h_by_Mpc})

        if self.method_non_linear_Pk in ['halofit', 'HALOFIT', 'Halofit', 'hmcode', 'Hmcode', 'HMcode', 'HMCODE']:
            self.need_cosmo_arguments(data, {'non linear': self.method_non_linear_Pk})
            print 'Using {:} to obtain the non-linear P(k, z)!'.format(self.method_non_linear_Pk)
            self.need_cosmo_arguments(data, {'halofit_tol_sigma':self.halofit_tol_sigma})
            print 'halofit tol sigma = {:}'.format(self.halofit_tol_sigma)
        else:
            print 'Only using the linear P(k, z) for ALL calculations \n (check keywords for "method_non_linear_Pk").'

        if self.method_non_linear_Pk in ['hmcode', 'Hmcode', 'HMcode', 'HMCODE']:
            # empirical value; might need revision...
            min_kmax_hmc = 170.
            if self.k_max_h_by_Mpc < min_kmax_hmc:
                self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': min_kmax_hmc})
                print "You're using HMcode, therefore increased k_max_h_by_Mpc to: {:.2f}".format(min_kmax_hmc)

        print 'Time for loading all data files:', time.time() - start_load   # Counting and printing time needed for loading data

        return

###  Alessio
############################################################
###  Reading in the p_z and the n(z)  ######################
############################################################
    def __load_redshift_distributions(self):


        #########################################################################
        ### Alessio. Loads the n(z)'s for foreground and cosmic shear samples ###
        ### Normalizes the imported distributions. Checked for cosmic shear. ####
        ### Checked for foregrounds too. ########################################
        #########################################################################

        ### Foreground samples

        self.pF = []
        self.pF_norm = np.zeros(self.nzbins_foreground, 'float64')
        self.z_max_foreground = np.zeros(self.nzbins_foreground, 'float64')  # Alessio: one z_max_foreground per each sample
        self.z_min_foreground = np.zeros(self.nzbins_foreground, 'float64')  # Alessio: one z_min_foreground per each sample
        self.redshifts_foreground = []
        self.spline_pz_foreground = []
        self.z_gg = []
        self.interp_nz_fg = []
        for sample_index in xrange(self.nzbins_foreground):

            fname_foreground = os.path.join(self.data_directory, 'red_for/nz_GAMAz{:}.dat'.format(sample_index+1))

            z_hist_foreground, n_z_hist_foreground = np.loadtxt(fname_foreground, unpack=True)
            self.redshifts_foreground.append(z_hist_foreground)
            self.pF.append(n_z_hist_foreground.tolist())

            #self.z_gg += [np.logspace(np.log10(z_hist_foreground.min()), np.log10(z_hist_foreground.max()), self.nzmax_gg[sample_index])]
            self.z_gg += [np.linspace(z_hist_foreground.min(), z_hist_foreground.max(), self.nzmax_gg[sample_index])]
            #self.z_gg += [np.linspace(1e-6, z_hist_foreground.max(), self.nzmax_gg[sample_index])]

            # normalize the distribution
            dz = self.redshifts_foreground[sample_index][1:] - self.redshifts_foreground[sample_index][:-1]
            self.pF_norm[sample_index] = np.sum(0.5 * (np.asarray(self.pF[sample_index][1:]) + np.asarray(self.pF[sample_index][:-1])) * dz)
            self.pF[sample_index][:] /= self.pF_norm[sample_index]
            nz_norm = np.sum(0.5 * (n_z_hist_foreground[1:] + n_z_hist_foreground[:-1]) * dz)
            self.z_max_foreground[sample_index] = self.redshifts_foreground[sample_index][:].max()
            self.z_min_foreground[sample_index] = self.redshifts_foreground[sample_index][:].min()

            # FK: we only need to define this once...
            self.spline_pz_foreground.append(itp.splrep(self.redshifts_foreground[sample_index], self.pF[sample_index][:]))

            self.interp_nz_fg.append(itp.interp1d(z_hist_foreground, n_z_hist_foreground / nz_norm, kind=self.type_interp_nz))

        ### Cosmic shear samples
        self.pz = np.zeros((self.nzmax_hist_background, self.nzbins_background), 'float64')
        self.pz_norm = np.zeros(self.nzbins_background, 'float64')
        self.spline_pz = []

        for zbin in xrange(self.nzbins_background):

            fname = os.path.join(self.data_directory, 'red_cs/nz_z{:}_kids_binned.dat'.format(zbin+1))

            z_hist, n_z_hist = np.loadtxt(fname, unpack=True)
            self.redshifts = z_hist

            self.pz[:, zbin] = n_z_hist
            self.spline_pz.append(itp.splrep(z_hist, n_z_hist))

            # normalize the distribution
            dz = self.redshifts[1:] - self.redshifts[:-1]
            self.pz_norm[zbin] = np.sum(0.5 * (self.pz[1:, zbin] + self.pz[:-1, zbin]) * dz)
            self.pz[:, zbin] /= self.pz_norm[zbin]

        self.z_max = self.redshifts.max()    #all bins for cosmic shear have the same min and max z's
        self.z_min = self.redshifts.min()

        # FK: define z-arrays for inner and outer integrations (MM and GM:
        #self.z_mm_out = np.linspace(self.z_min, self.z_max, self.nzmax_mm_out)
        self.z_mm_out = np.logspace(np.log10(self.z_min), np.log10(self.z_max), self.nzmax_mm_out)

        #message for Fabian: the grid defined below is exactly the one that Edo uses for his trapezoidal integration
        # in our case using this grid does not give so much better results, I think because we would still need to use the
        # Romberg integration for the inner integrations. Anyway I leave it here, in case you want to experiment with it

        #self.z_mm_out = np.zeros(self.nzmax_mm_out, 'float64')
        #self.z_mm_out[0] = 0.001
        #self.z_mm_out[1] = 0.025
        #for i in range(2, 36):
        #    self.z_mm_out[i] = self.z_mm_out[i-1] + 0.154*self.z_mm_out[i-1]
        #self.z_mm_out[36] = 3.475

        self.z_mm_in = np.linspace(self.z_min, self.z_max, self.nzmax_mm_in)
        # currently set to range of MM: --> NO, range of foreground
        self.z_gm_out = np.logspace(np.log10(min(self.z_min_foreground)), np.log10(max(self.z_max_foreground)), self.nzmax_gm_out)
        #self.z_gm_out = np.logspace(np.log10(self.z_min), np.log10(self.z_max), self.nzmax_gm_out)
        #self.z_gm_out = np.linspace(self.z_min, self.z_max, self.nzmax_gm_out)
        self.z_gm_in = np.linspace(self.z_min, self.z_max, self.nzmax_gm_in)
        # defined now in foregorund loop!
        #self.z_gg = np.linspace(min(self.z_min_foreground), max(self.z_max_foreground), self.nzmax_gg)
        #self.z_gg = np.logspace(np.log10(min(self.z_min_foreground)), np.log10(max(self.z_max_foreground)), self.nzmax_gg)

        return


    # FK: function to load data-vector:
    def __load_data_vector(self):

        fname = os.path.join(self.data_directory, 'Pkids_data_full_ibc2.dat')
        ell_bins, data_vec = np.loadtxt(fname, unpack=True)
        print 'Loaded data vector from: \n', fname

        return ell_bins, data_vec

    # FK: function to load covariance matrix:
    def __load_covariance_matrix(self):

        # try to load the actual 100x100 matrix first:
        try:
            fname = os.path.join(self.data_directory, 'cov_matrix_ana.dat')
            cov = np.loadtxt(fname)
            print 'Loaded covariance matrix from: \n', fname

        # if that doesn't exist yet, create it from Edo's original data:
        except:
            fname = os.path.join(self.data_directory, 'Pkidscov_data_full2_iter2.dat')
            data = np.loadtxt(fname)
            print 'Loaded list for covariance matrix from: \n', fname
            cov = self.__get_matrix_from_list(data)
            fname = os.path.join(self.data_directory, 'cov_matrix_ana.dat')
            np.savetxt(fname, cov)
            print 'Saved covariance matrix to: \n', fname

        return cov

    # FK: helper function to create matrix from Edo's list:
    def __get_matrix_from_list(self, data_as_list):

        dim_data = len(data_as_list)
        dim_ax = int(np.sqrt(dim_data))

        # really simple and works for us:
        #cov = data[:, -1].reshape(dim_ax, dim_ax)

        # the much safer and most general solution (unpractical for very large matrices)
        cov = np.zeros((dim_ax, dim_ax))
        for idx1 in xrange(dim_ax):
            for idx2 in xrange(dim_ax):

                # this is quite costly, but covers the most general index-case:
                for idx_lin in xrange(dim_data):

                    if idx1 + 1 == data_as_list[idx_lin, 0] and idx2 + 1 == data_as_list[idx_lin, 1]:
                        cov[idx1, idx2] = data_as_list[idx_lin, -1]

        return cov

    # FK: get masking indices:
    def __get_masking_indices(self):

        for idx_zcorrs in xrange(self.nzcorrs_mm):
            if idx_zcorrs == 0:
                bands_to_use_for_MM = self.bands_EE_to_use_MM
            else:
                bands_to_use_for_MM = np.concatenate((bands_to_use_for_MM, self.bands_EE_to_use_MM))

        for idx_zcorrs in xrange(self.nzcorrs_gm):
            if idx_zcorrs == 0:
                bands_to_use_for_GM = self.bands_EE_to_use_GM
            else:
                bands_to_use_for_GM = np.concatenate((bands_to_use_for_GM, self.bands_EE_to_use_GM))

        for idx_zbin in xrange(self.nzbins_foreground):
            if idx_zbin == 0:
                bands_to_use_for_GG = self.bands_EE_to_use_GG
            else:
                bands_to_use_for_GG = np.concatenate((bands_to_use_for_GG, self.bands_EE_to_use_GG))
        bands_to_use = np.concatenate((bands_to_use_for_MM, bands_to_use_for_GM, bands_to_use_for_GG))

        index1 = np.where(bands_to_use == 1)[0]

        return index1

###  Alessio
###############################################################################
###  Actual calculation of the loglk:      ####################################
###  theory vector is calculated below     ####################################
###############################################################################
    def loglkl(self, cosmo, data):

        # FK:
        # skip the calculations if all flags are 0 for a probe:
        if self.flag_no_MM:
            band_powers_theory_matter_matter = np.zeros(self.nzcorrs_mm * self.nellsmax)
        else:
            band_powers_theory_matter_matter = self.calculate_theory_matter_matter(cosmo, data)

        if self.flag_no_GM:
            band_powers_theory_galaxy_matter = np.zeros(self.nzcorrs_gm * self.nellsmax)
        else:
            band_powers_theory_galaxy_matter = self.calculate_theory_galaxy_matter(cosmo, data)

        if self.flag_no_GG:
            band_powers_theory_galaxy_galaxy = np.zeros(self.nzbins_foreground * self.nellsmax)
        else:
            band_powers_theory_galaxy_galaxy = self.calculate_theory_galaxy_galaxy(cosmo, data)

        # FK: make one joint theory vector:
        band_powers_theory = np.concatenate((band_powers_theory_matter_matter, band_powers_theory_galaxy_matter, band_powers_theory_galaxy_galaxy))
        #print band_powers_theory.shape
        #print self.band_powers.shape

        difference_vector = self.band_powers - band_powers_theory
        difference_vector = difference_vector[self.mask_indices]

        # Don't invert that matrix!
        #chi2 = difference_vector.T.dot(inv_cov_sliced.dot(difference_vector))
        # this is for running smoothly with MultiNest
        # (in initial checking of prior space, there might occur weird solutions)
        if np.isinf(band_powers_theory).any() or np.isnan(band_powers_theory).any():
            chi2 = 2e12
        else:
            # use a Cholesky decomposition instead:
            yt = solve_triangular(self.cholesky_transform, difference_vector, lower=True)
            chi2 = yt.dot(yt)

        #print 'Chi2 = {:.4f}'.format(chi2)

        return -0.5 * chi2

    # FK: helper function for MM, GM, and GG
    def get_matter_power_spectrum(self, r, z, ells, nzmax, nellsmax, cosmo, data):

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        # this doesn't really have to go into the loop over fields!
        pk = np.zeros((nellsmax, nzmax),'float64')
        k_max_in_inv_Mpc = self.k_max_h_by_Mpc * cosmo.h()
        for index_ells in xrange(nellsmax):
            for index_z in xrange(nzmax):
                # standard Limber approximation:
                #k = ells[index_ells] / r[index_z]
                # extended Limber approximation (cf. LoVerde & Afshordi 2008):
                k_in_inv_Mpc = (ells[index_ells] + 0.5) / r[index_z]
                if k_in_inv_Mpc > k_max_in_inv_Mpc:
                    pk_dm = 0.
                else:
                    pk_dm = cosmo.pk(k_in_inv_Mpc, z[index_z])
                #pk[index_ells,index_z] = cosmo.pk(ells[index_ells]/r[index_z], self.redshifts[index_z])
                """
                if self.baryon_feedback:
                    if 'A_bary' in data.mcmc_parameters:
                        A_bary = data.mcmc_parameters['A_bary']['current'] * data.mcmc_parameters['A_bary']['scale']
                        #print 'A_bary={:.4f}'.format(A_bary)
                        pk[index_ells, index_z] = pk_dm * self.baryon_feedback_bias_sqr(k, z[index_z], A_bary=A_bary)
                    else:
                        pk[index_ells, index_z] = pk_dm * self.baryon_feedback_bias_sqr(k, z[index_z])
                else:
                    pk[index_ells, index_z] = pk_dm
                """

                if 'A_bary' in data.mcmc_parameters:
                    A_bary = data.mcmc_parameters['A_bary']['current'] * data.mcmc_parameters['A_bary']['scale']
                    pk[index_ells, index_z] = pk_dm * self.baryon_feedback_bias_sqr(k_in_inv_Mpc / cosmo.h(), z[index_z], A_bary=A_bary)
                    # don't apply the baryon feedback model to the linear Pk!
                    #self.pk_lin[index_ell, index_z] = pk_lin_dm * self.baryon_feedback_bias_sqr(k_in_inv_Mpc / self.small_h, self.z_p[index_z], A_bary=A_bary)
                else:
                    pk[index_ells, index_z] = pk_dm
                    #pk_lin[index_ells, index_z] = pk_lin_dm


        # Define the alpha function, that will characterize the theoretical
        # uncertainty. Chosen to be 0.001 at low k, raise between 0.1 and 0.2
        # to self.theoretical_error
        if 'nonlinear_scales_nuisance' in data.mcmc_parameters:

            # Recover the non_linear scale computed by halofit. If no scale was
            # affected, set the scale to one, and make sure that the nuisance
            # parameter epsilon is set to zero
            k_sigma = np.zeros(nzmax, 'float64')
            if (cosmo.nonlinear_method == 0):
                k_sigma[:] = 1.e6
            else:
                k_sigma = cosmo.nonlinear_scale(z, nzmax)

            # replace unphysical values of k_sigma
            if not (cosmo.nonlinear_method == 0):
                k_sigma_problem = False
                for index_z in xrange(nzmax-1):
                    if (k_sigma[index_z+1]<k_sigma[index_z]) or (k_sigma[index_z+1]>2.5):
                        k_sigma[index_z+1] = 2.5
                        k_sigma_problem = True
                    #print("%s\t%s" % (k_sigma[index_z], self.z[index_z]))
                if k_sigma_problem:
                    warnings.warn("There were unphysical (decreasing in redshift or exploding) values of k_sigma (=cosmo.nonlinear_scale(...)). To proceed they were set to 2.5, the highest scale that seems to be stable.")

            alpha = np.zeros((nellsmax, nzmax), 'float64')
            nonlinear_scales_nuisance = data.mcmc_parameters['nonlinear_scales_nuisance']['current'] * data.mcmc_parameters['nonlinear_scales_nuisance']['scale']
            for index_l in xrange(nellsmax):
                for index_z in xrange(1, nzmax):
                    k = (ells[index_l]+0.5)/r[index_z] # Limber's approx. LoVerde-Afshordi
                    if k > k_sigma[index_z]:
                        alpha[index_l, index_z] = np.log(1. + k / k_sigma[index_z]) * nonlinear_scales_nuisance

            np.multiply(pk, 1.+alpha)

        return pk

    def get_matter_power_spectrum_for_GG(self, k, z, cosmo, data):

        k_max_in_inv_Mpc = self.k_max_h_by_Mpc * cosmo.h()

        if k > k_max_in_inv_Mpc:
            pk_dm = 0.
        else:
            pk_dm = cosmo.pk(k, z)

        """
        if self.baryon_feedback:
            if 'A_bary' in data.mcmc_parameters:
            #if 'A_bary' in self.nuisance_params:
                A_bary = data.mcmc_parameters['A_bary']['current'] * data.mcmc_parameters['A_bary']['scale']
                #A_bary = self.nuisance_params['A_bary']
                #print 'A_bary={:.4f}'.format(A_bary)
                pk = pk_dm * self.baryon_feedback_bias_sqr(k, z, A_bary=A_bary)
            else:
                pk = pk_dm * self.baryon_feedback_bias_sqr(k, z)
        else:
            pk = pk_dm
        """

        if 'A_bary' in data.mcmc_parameters:
            A_bary = data.mcmc_parameters['A_bary']['current'] * data.mcmc_parameters['A_bary']['scale']
            pk = pk_dm * self.baryon_feedback_bias_sqr(k / cosmo.h(), z, A_bary=A_bary)
        else:
            pk = pk_dm

        return pk

    # FK: helper function for MM and GM
    def get_lensing_kernel_g_of_r(self, r_in, r_out, z_in, pr_in, nzmax_in, nzmax_out, nzbins):

        # FK: precompute g(r) at inner grid resolution and spline it.
        # Compute function g_i(r), that depends on r and the bin
        # g_i(r) = (1+z(r)) int_r^+\infty drs eta_r(rs) (rs-r)/rs
        g_tmp = np.zeros((nzmax_in, nzbins), 'float64')
        #g = np.zeros((len(self.redshifts), self.nzbins_background), 'float64')
        for zbin in xrange(nzbins):
            for nr in xrange(nzmax_in - 1):
            #for nr in xrange(1, len(self.redshifts) - 1):
                fun = pr_in[nr:, zbin] * (r_in[nr:] - r_in[nr]) / r_in[nr:]
                g_tmp[nr, zbin] = np.sum(0.5 * (fun[1:] + fun[:-1])* (r_in[nr + 1:] - r_in[nr:-1])) # Alessio * (r[nr + 1:] - r[nr:-1]))
                g_tmp[nr, zbin] *= r_in[nr] * (1. + z_in[nr])   # Alessio: modified weight function
                #g[nr, zbin] *= r[nr]*(1. + self.redshifts[nr])   # Alessio: modified weight function

        # FK: get samples of g(r) at resolution of outer grid:
        g = np.zeros((nzmax_out, nzbins), 'float64')
        for zbin in xrange(nzbins):
            spline_g = itp.splrep(np.log(r_in), g_tmp[:, zbin])
            g[:, zbin] = itp.splev(np.log(r_out), spline_g)

        return g

### Alessio
###############################################################################
### Calculation of the theoretical power spectra, for COSMIC SHEAR only
### theoretical covariances and m corr ########################################
###############################################################################
    def calculate_theory_matter_matter(self, cosmo, data):

        Omega_m = cosmo.Omega_m()
        small_h = cosmo.h()

        ### Alessio
        ###################################
        ### Intrinsic alignments ##########
        ###################################

        # needed for IA modelling:
        if ('A_IA' in data.mcmc_parameters) and ('exp_IA' in data.mcmc_parameters):
            amp_IA = data.mcmc_parameters['A_IA']['current'] * data.mcmc_parameters['A_IA']['scale']
            exp_IA = data.mcmc_parameters['exp_IA']['current'] * data.mcmc_parameters['exp_IA']['scale']
            intrinsic_alignment = True
        elif ('A_IA' in data.mcmc_parameters) and ('exp_IA' not in data.mcmc_parameters):
            amp_IA = data.mcmc_parameters['A_IA']['current'] * data.mcmc_parameters['A_IA']['scale']
            # redshift-scaling is turned off:
            exp_IA = 0.
            intrinsic_alignment = True
        else:
            intrinsic_alignment = False




        ### Alessio
        ###################################
        ### Distance-redshift relation ####
        ###################################


        # get distances from cosmo-module:
        # FK: use outer and inner grid:
        r_out, dzdr_out = cosmo.z_of_r(self.z_mm_out)
        r_in, dzdr_in = cosmo.z_of_r(self.z_mm_in)

        ### Alessio
        ###################################
        ### ell-modes ###########
        ###################################

        ells = self.ell_bin_centers

        ### Alessio
        #####################################
        ### Matter Power Spectrum ###########
        #####################################

        pk = self.get_matter_power_spectrum(r_out, self.z_mm_out, ells, self.nzmax_mm_out, self.nellsmax, cosmo, data)

        screening_scale = pow(10.,data.mcmc_parameters['screening_scale']['current'])
        #screening_scale = data.mcmc_parameters['screening_scale']['current']
        if intrinsic_alignment:         ### Growth factor is scale dependent in MG! and we perform Limber's approx, k = (ell+0.5)/z
            rho_crit = self.get_critical_density(small_h)
            # derive the linear growth factor D(z)
            #linear_growth_rate = np.zeros_like(self.z_mm_out)
            linear_growth_rate = np.zeros((self.nellsmax,self.nzmax_mm_out), 'float64')
            #print self.redshifts
            for index_ell in range(self.nellsmax):
            	for index_z, z in enumerate(self.z_mm_out):
                #try:
                    # for CLASS ver >= 2.6:
                    #linear_growth_rate[index_z] = cosmo.scale_independent_growth_factor(z)
                    k = (ells[index_ell]+0.5)/r_out[index_z]
                    linear_growth_rate[index_ell][index_z] = cosmo.screened_growth(k, z, screening_scale)
                    #qui inserire screened growth
	



        ### Alessio
        ############################################
        ### Redshift distributions normalization ###
        ############################################

        # FK: background n(z):
        # FK: removed unnecessary lists...
        pr_in = np.zeros((self.nzmax_mm_in, self.nzbins_background))
        pr_out = np.zeros((self.nzmax_mm_out, self.nzbins_background))
        for zbin in range(self.nzbins_background):
            pz = itp.splev(self.z_mm_in, self.spline_pz[zbin])
            pr_in[:, zbin] = pz * dzdr_in
            pz = itp.splev(self.z_mm_out, self.spline_pz[zbin])
            pr_out[:, zbin] = pz * dzdr_out

        ### Alessio
        ######################################################################
        ### Actual calculation of theoretical cosmic shear power spectrum  ###
        ######################################################################

        # FK: get lensing kernel:
        g = self.get_lensing_kernel_g_of_r(r_in, r_out, self.z_mm_in, pr_in, self.nzmax_mm_in, self.nzmax_mm_out, self.nzbins_background)

        # Start loop over l for computation of P_l^shear
        Pl_GG_integrand = np.zeros((self.nzmax_mm_out, self.nzbins_background, self.nzbins_background), 'float64')
        #Pl_GG_integrand = np.zeros((len(self.redshifts), self.nzbins_background, self.nzbins_background), 'float64')
        Pl_GG = np.zeros((self.nellsmax, self.nzbins_background, self.nzbins_background), 'float64')
        if intrinsic_alignment:
            Pl_II_integrand = np.zeros_like(Pl_GG_integrand)
            Pl_II = np.zeros_like(Pl_GG)

            Pl_GI_integrand = np.zeros_like(Pl_GG_integrand)
            Pl_GI = np.zeros_like(Pl_GG)

        ell_norm = np.asarray(ells) * np.asarray(ells) / (2. * np.pi)

        dr = r_out[1:] - r_out[:-1]

        ## MG part for calculation of spectra starts here
        H0squared = cosmo.get_Hubble(0.)*cosmo.get_Hubble(0.)
        poissonfactor = H0squared*Omega_m
        scalefac = np.zeros(self.nzmax_mm_out, 'float64')
        for index_z in xrange(self.nzmax_mm_out):
             scalefac[index_z] = 1./(1.+self.z_mm_out[index_z])
        #screening_scale = data.mcmc_parameters['screening_scale']['current']
        for index_ell in xrange(self.nellsmax):
            mu = np.zeros(self.nzmax_mm_out, 'float64')
            gamma = np.zeros(self.nzmax_mm_out, 'float64')
            #mu = np.zeros((self.nzmax_mm_out, self.nzmax_mm_out), 'float64')
            #gamma = np.zeros((self.nzmax_mm_out,self.nzmax_mm_out), 'float64')
            #kvec = np.zeros((self.nellsmax, self.nzmax_mm_out), 'float64')
            for index_z in xrange(self.nzmax_mm_out):
                # standard Limber approximation:
                #k = ells[index_ells] / r[index_z]
                # extended Limber approximation (cf. LoVerde & Afshordi 2008):
                k = (ells[index_ell] + 0.5) / r_out[index_z]
                a = (-k*k)*(2./3.)*scalefac[index_z]/poissonfactor
                mu[index_z] = cosmo.get_poissonratio_screened(k, self.z_mm_out[index_z], screening_scale, a) #*(-k*k)*(2./3.)*scalefac[index_z]/poissonfactor
                gamma[index_z] = cosmo.get_phipsiratio_screened(k, self.z_mm_out[index_z], screening_scale)
                # find Pl_integrand = (g(r))**2 * P(l/r,z(r))
            for zbin1 in xrange(self.nzbins_background):
                for zbin2 in xrange(zbin1, self.nzbins_background): #self.nzbins_background):
                    Pl_GG_integrand[:, zbin1, zbin2] = g[:, zbin1] * g[:, zbin2] * pk[index_ell, :]  / r_out**2 *(1.+1./gamma[:])*(1.+1./gamma[:])*mu[:] *mu[:]/4. # Alessio: modified accordingly to modified weight above

                    if intrinsic_alignment:
                        factor_IA = self.get_factor_IA(self.z_mm_out[:], linear_growth_rate[index_ell][:], rho_crit, small_h, Omega_m, amp_IA, exp_IA)
                        #factor_IA = self.get_factor_IA(self.z_mm_out[:], linear_growth_rate[:], rho_crit, small_h, Omega_m, amp_IA, exp_IA)
                        #print F_of_x
                        #print self.eta_r[1:, zbin1].shape

                        Pl_II_integrand[:, zbin1, zbin2] = pr_out[:, zbin1] * pr_out[:, zbin2] * factor_IA**2 / r_out**2 * pk[index_ell, :] *mu[:] *mu[:]
                        Pl_GI_integrand[:, zbin1, zbin2] = (g[:, zbin1] * pr_out[:, zbin2] + g[:, zbin2] * pr_out[:, zbin1]) * factor_IA * pk[index_ell, :] / r_out**2 *(1.+1./gamma[:])*mu[:]*mu[:]/2.

            # Integrate over r to get  P_ij(l)^shear
            # P_ij(l)^shear = 9/4 Omega0_m^2 H_0^4 \sum_0^rmax dr (g_i(r) g_j(r)) P(k=l/r,z(r))
            for zbin1 in xrange(self.nzbins_background):
                for zbin2 in xrange(zbin1, self.nzbins_background): #self.nzbins_background):
                    Pl_GG[index_ell, zbin1, zbin2] = np.sum(0.5 * (Pl_GG_integrand[1:, zbin1, zbin2] + Pl_GG_integrand[:-1, zbin1, zbin2]) * dr)
                    # here we divide by 16, because we get a 2^2 from g(z)!  # Alessio: modified this accordingly to what previously modified
                    Pl_GG[index_ell, zbin1, zbin2] *= 9. / 4. * Omega_m**2 # in units of Mpc**4
                    Pl_GG[index_ell, zbin1, zbin2] *= (small_h / 2997.9)**4 # dimensionless
                    Pl_GG[index_ell, zbin1, zbin2] *= ell_norm[index_ell]

                    if intrinsic_alignment:
                        Pl_II[index_ell, zbin1, zbin2] = np.sum(0.5 * (Pl_II_integrand[1:, zbin1, zbin2] + Pl_II_integrand[:-1, zbin1, zbin2]) * dr)
                        Pl_II[index_ell, zbin1, zbin2] *= ell_norm[index_ell]

                        Pl_GI[index_ell, zbin1, zbin2] = np.sum(0.5 * (Pl_GI_integrand[1:, zbin1, zbin2] + Pl_GI_integrand[:-1, zbin1, zbin2]) * dr)
                        # here we divide by 4, because we get a 2 from g(r)!   # Alessio: modified accordingly to what previously modified
                        Pl_GI[index_ell, zbin1, zbin2] *= 3. / 2. * Omega_m
                        Pl_GI[index_ell, zbin1, zbin2] *= (small_h / 2997.9)**2
                        Pl_GI[index_ell, zbin1, zbin2] *= ell_norm[index_ell]

        if intrinsic_alignment:
            Pl = Pl_GG + Pl_GI + Pl_II
        else:
            Pl = Pl_GG



        #for zbin1 in xrange(self.nzbins_background):
        #    for zbin2 in xrange(self.nzbins_background):

        #        np.savetxt(self.path_save + 'P_MM_theory_B{:}B{:}.txt'.format(zbin1, zbin2), Pl[:, zbin1, zbin2])
                # FK:
        #        if intrinsic_alignment:
        #            np.savetxt(self.path_save + 'P_MM_theory_GI_B{:}B{:}.txt'.format(zbin1, zbin2), Pl_GI[:, zbin1, zbin2])
        #            np.savetxt(self.path_save + 'P_MM_theory_II_B{:}B{:}.txt'.format(zbin1, zbin2), Pl_II[:, zbin1, zbin2])


        # FK: we still need to sort out the unique elements!
        #band_powers_theory_cosmic_shear = Pl.flatten()
        band_powers_theory = np.zeros((self.nellsmax * self.nzcorrs_mm))
        idx_lin = 0
        for zbin1 in xrange(self.nzbins_background):
            for zbin2 in xrange(zbin1, self.nzbins_background):
                for idx_ell in xrange(self.nellsmax):
                    band_powers_theory[idx_lin] = Pl[idx_ell, zbin1, zbin2]
                    idx_lin += 1

        return band_powers_theory


##########################################################################################################################################################
##########################################################################################################################################################

### Alessio
####################################################################################
### Calculation of the theoretical power spectra, for galaxy - matter
### theoretical covariances and m corr #############################################
####################################################################################
    def calculate_theory_galaxy_matter(self, cosmo, data):

        Omega_m = cosmo.Omega_m()
        small_h = cosmo.h()

        ### Alessio
        ###################################
        ### Distance-redshift relation ####
        ###################################

        # get distances from cosmo-module:
        r_out, dzdr_out = cosmo.z_of_r(self.z_gm_out)
        r_in, dzdr_in = cosmo.z_of_r(self.z_gm_in)

        # FK:
        #if ('bias' in data.mcmc_parameters):
        #    bias = data.mcmc_parameters['bias']['current']*data.mcmc_parameters['bias']['scale']
        bias = np.zeros(self.nzbins_foreground, 'float64')
        for sample_index in range(self.nzbins_foreground):
            if ('bias_z{:}'.format(sample_index + 1) in data.mcmc_parameters):
                bias[sample_index] = data.mcmc_parameters['bias_z{:}'.format(sample_index + 1)]['current'] * data.mcmc_parameters['bias_z{:}'.format(sample_index + 1)]['scale']

        ### Alessio
        ###################################
        ### Intrinsic alignment ###########
        ###################################

        # needed for IA modelling:
        if ('A_IA' in data.mcmc_parameters) and ('exp_IA' in data.mcmc_parameters):
            amp_IA = data.mcmc_parameters['A_IA']['current'] * data.mcmc_parameters['A_IA']['scale']
            exp_IA = data.mcmc_parameters['exp_IA']['current'] * data.mcmc_parameters['exp_IA']['scale']
            intrinsic_alignment = True
        elif ('A_IA' in data.mcmc_parameters) and ('exp_IA' not in data.mcmc_parameters):
            amp_IA = data.mcmc_parameters['A_IA']['current'] * data.mcmc_parameters['A_IA']['scale']
            # redshift-scaling is turned off:
            exp_IA = 0.

            intrinsic_alignment = True
        else:
            intrinsic_alignment = False


        # FK: background n(z):
        # FK: removed unnecessary lists...
        pr_in = np.zeros((self.nzmax_gm_in, self.nzbins_background))
        pr_out = np.zeros((self.nzmax_gm_out, self.nzbins_background))
        for zbin in range(self.nzbins_background):
            pz = itp.splev(self.z_gm_in, self.spline_pz[zbin])
            pr_in[:, zbin] = pz * dzdr_in
            pz = itp.splev(self.z_gm_out, self.spline_pz[zbin])
            pr_out[:, zbin] = pz * dzdr_out

        # FK: foreground n(z):
        # TODO: simplify further...
        interp_pz_foreground = np.zeros((self.nzmax_gm_out, self.nzbins_foreground),'float64')
        pr_foreground = np.zeros((self.nzmax_gm_out, self.nzbins_foreground),'float64')
        for zbin in xrange(self.nzbins_foreground):
            mask1 = self.z_gm_out >= self.z_min_foreground[zbin]
            mask2 = self.z_gm_out <= self.z_max_foreground[zbin]
            mask = mask1 & mask2
            interp_pz_foreground[mask, zbin] = itp.splev(self.z_gm_out[mask], self.spline_pz_foreground[zbin])
            pr_foreground[:, zbin] = interp_pz_foreground[:, zbin] * dzdr_out

        ### Alessio
        ###################################
        ### ell-modes ###########
        ###################################

        ells = self.ell_bin_centers
        #nellsmax = len(self.ell_bin_centers)
        screening_scale = pow(10.,data.mcmc_parameters['screening_scale']['current'])
        #screening_scale = data.mcmc_parameters['screening_scale']['current']
        
        if intrinsic_alignment:
            rho_crit = self.get_critical_density(small_h)
            # derive the linear growth factor D(z)
            #linear_growth_rate = np.zeros_like(self.z_gm_out)
            linear_growth_rate = np.zeros((self.nellsmax,self.nzmax_gm_out), 'float64')
            #print self.redshifts
            for index_ell in range(self.nellsmax):
                for index_z, z in enumerate(self.z_gm_out):
                #try:
                    # for CLASS ver >= 2.6:
                    #linear_growth_rate[index_z] = cosmo.scale_independent_growth_factor(z)
                    k = (ells[index_ell]+0.5)/r_out[index_z]
                    linear_growth_rate[index_ell][index_z] = cosmo.screened_growth(k, z, screening_scale)
                    
        ### Alessio
        #####################################
        ### Matter Power Spectrum ###########
        #####################################

        pk = self.get_matter_power_spectrum(r_out, self.z_gm_out, ells, self.nzmax_gm_out, self.nellsmax, cosmo, data)

        ### Alessio
        ######################################################################
        ### Actual calculation of theoretical galaxy- matter power spectrum  ###
        ######################################################################

        # FK: get lensing kernel:
        g = self.get_lensing_kernel_g_of_r(r_in, r_out, self.z_gm_in, pr_in, self.nzmax_gm_in, self.nzmax_gm_out, self.nzbins_background)

        # Start loop over l for computation of C_l^shear
        Pl_GM_integrand = np.zeros((self.nzmax_gm_out, self.nzbins_foreground, self.nzbins_background), 'float64')
        Pl_GM = np.zeros((self.nellsmax, self.nzbins_foreground, self.nzbins_background), 'float64')

        if intrinsic_alignment:

            Pl_gI_integrand = np.zeros_like(Pl_GM_integrand)
            Pl_gI = np.zeros_like(Pl_GM)

        ell_norm = np.asarray(ells) * np.asarray(ells)/ (2. * np.pi)

        dr = r_out[1:] - r_out[:-1]

         ## MG part for calculation of spectra starts here
        H0squared = cosmo.get_Hubble(0.)*cosmo.get_Hubble(0.)
        poissonfactor = H0squared*Omega_m
        scalefac = np.zeros(self.nzmax_gm_out, 'float64')
        for index_z in xrange(self.nzmax_gm_out):
             scalefac[index_z] = 1./(1.+self.z_gm_out[index_z])
        #screening_scale = data.mcmc_parameters['screening_scale']['current']
        # FK: removed shifts like array[1:, ...]; only necessary if z[0]=0!
        for index_ell in xrange(self.nellsmax):
            mu = np.zeros(self.nzmax_gm_out, 'float64')
            gamma = np.zeros(self.nzmax_gm_out, 'float64')
            #mu = np.zeros((self.nzmax_mm_out, self.nzmax_gm_out), 'float64')
            #gamma = np.zeros((self.nzmax_mm_out,self.nzmax_gm_out), 'float64')
            #kvec = np.zeros((self.nellsmax, self.nzmax_gm_out), 'float64')
            for index_z in xrange(self.nzmax_gm_out):
                # standard Limber approximation:
                #k = ells[index_ells] / r[index_z]
                # extended Limber approximation (cf. LoVerde & Afshordi 2008):
                k = (ells[index_ell] + 0.5) / r_out[index_z]
                a = (-k*k)*(2./3.)*scalefac[index_z]/poissonfactor
                mu[index_z] = cosmo.get_poissonratio_screened(k, self.z_gm_out[index_z], screening_scale, a) *(-k*k)*(2./3.)*scalefac[index_z]/poissonfactor
                gamma[index_z] = cosmo.get_phipsiratio_screened(k, self.z_gm_out[index_z], screening_scale)
            # find Pl_integrand = (g(r))**2 * P(l/r,z(r))
            for sample_index in xrange(self.nzbins_foreground):
                for zbin1 in xrange(self.nzbins_background):
                    Pl_GM_integrand[:, sample_index, zbin1] = g[:, zbin1] * pr_foreground[:, sample_index] * pk[index_ell, :] / r_out**2 *(1.+1./gamma[:])*mu[:]*mu[:]/2.  # Alessio: modified accordingly to modified weight above
                    #print "GM", Pl_GM_integrand[1:, zbin1, sample_index]
                    if intrinsic_alignment:
                        factor_IA = self.get_factor_IA(self.z_gm_out[:], linear_growth_rate[index_ell][:], rho_crit, small_h, Omega_m, amp_IA, exp_IA)
                        #factor_IA = self.get_factor_IA(self.z_gm_out[:], linear_growth_rate[:], rho_crit, small_h, Omega_m, amp_IA, exp_IA)
                        #print F_of_x
                        #print self.eta_r[1:, zbin1].shape

                        Pl_gI_integrand[:, sample_index, zbin1] = pr_out[:, zbin1] * pr_foreground[:, sample_index] * factor_IA * bias[sample_index] / r_out**2 * pk[index_ell, :]#*mu[:]*mu[:]

            # Integrate over r to get P_ij(l)^galaxy-matter
            # P_ij(l)^galaxy-matter = 9/4 Omega0_m^2 H_0^4 \sum_0^rmax dr (g_i(r) g_j(r)) P(k=l/r,z(r))
            for sample_index in xrange(self.nzbins_foreground):
                for zbin1 in xrange(self.nzbins_background):
                    Pl_GM[index_ell, sample_index, zbin1] = np.sum(0.5 * (Pl_GM_integrand[1:, sample_index, zbin1] + Pl_GM_integrand[:-1, sample_index, zbin1]) * dr)
                    # here we divide by 16, because we get a 2^2 from g(z)!  # Alessio: modified this accordingly to what previously modified
                    Pl_GM[index_ell, sample_index, zbin1] *= 3. / 2. * Omega_m # in units of Mpc**2
                    Pl_GM[index_ell, sample_index, zbin1] *= (small_h / 2997.9)**2 # dimensionless
                    Pl_GM[index_ell, sample_index, zbin1] *= bias[sample_index]
                    Pl_GM[index_ell, sample_index, zbin1] *= ell_norm[index_ell]

                    if intrinsic_alignment:
                        Pl_gI[index_ell, sample_index, zbin1] = np.sum(0.5 * (Pl_gI_integrand[1:, sample_index, zbin1] + Pl_gI_integrand[:-1, sample_index, zbin1]) * dr)
                        Pl_gI[index_ell, sample_index, zbin1] *= ell_norm[index_ell]
                        # here we divide by 4, because we get a 2 from g(r)!
                        #Pl_gI[index_ell, sample_index, zbin1] *= 3. / 2. * Omega_m
                        #Pl_gI[index_ell, sample_index, zbin1] *= (small_h / 2997.9)**2

        if intrinsic_alignment:
            Pl = Pl_GM + Pl_gI
        else:
            Pl = Pl_GM

        #for sample_index in xrange(self.nzbins_foreground):
        #    for zbin1 in xrange(self.nzbins_background):
        #np.savetxt(self.path_save + 'P_GM_theory_F{:}B{:}.txt'.format(sample_index, zbin1), Pl[:, sample_index, zbin1])
        #        if intrinsic_alignment:
        #            np.savetxt(self.path_save + 'P_GM_theory_F{:}B{:}.txt'.format(sample_index, zbin1), Pl_GM[:, sample_index, zbin1] + Pl_gI[:, sample_index, zbin1])
        #            np.savetxt(self.path_save + 'P_GM_theory_gI_F{:}B{:}.txt'.format(sample_index, zbin1), Pl_gI[:, sample_index, zbin1])
        #        else:
        #            np.savetxt(self.path_save + 'P_GM_theory_F{:}B{:}.txt'.format(sample_index, zbin1), Pl_GM[:, sample_index, zbin1])


        # we need to resort the vector:
        Pl_lin = np.zeros((self.nellsmax * self.nzbins_foreground * self.nzbins_background), 'float64')
        index_lin = 0
        for sample_index in xrange(self.nzbins_foreground):
            for zbin1 in xrange(self.nzbins_background):
                for index_ell in xrange(self.nellsmax):
                    Pl_lin[index_lin] = Pl[index_ell, sample_index, zbin1]
                    index_lin += 1

        return Pl_lin


##########################################################################################################################################################
##########################################################################################################################################################

### Alessio
####################################################################################
### Calculation of the theoretical power spectra, for galaxy-galaxy
### theoretical covariances and m corr #############################################
####################################################################################
    def calculate_theory_galaxy_galaxy(self, cosmo, data):

        Omega_m = cosmo.Omega_m()
        small_h = cosmo.h()

        ells = self.ell_bin_centers
        ell_norm = np.asarray(ells) * np.asarray(ells) / (2. * np.pi)

        # we're only interested in the auto-correlations here!
        Pl_GG = np.zeros((self.nellsmax * self.nzbins_foreground), 'float64')

        #print np.arange(0.035, 0.195, 0.01)
        self.z_gg[0][1:17] = np.arange(0.035, 0.195, 0.01)

        #self.z_gg[0][0] = 0.00001
        #self.z_gg[0][1] = 0.03
        #for i in range(3,17):
        #    self.z_gg[0][i-1] = 0.03 + (0.2-0.03)/15.*(i-2)
        #print self.z_gg[0]
        #print np.arange(0.035, 0.205, 0.01)
        #self.z_gg[0][1:17] = np.arange(0.035, 0.205, 0.01)

        #print np.arange(0.205, 0.495, 0.01)
        self.z_gg[1][0:30] = np.arange(0.205, 0.495, 0.01)

        index_lin = 0
        for zbin in xrange(self.nzbins_foreground):

            r, dzdr = cosmo.z_of_r(self.z_gg[zbin])

            ## MG part for calculation of spectra starts here
            H0squared = cosmo.get_Hubble(0.)*cosmo.get_Hubble(0.)
            poissonfactor = H0squared*Omega_m
            scalefac = np.zeros(self.nzmax_gg[zbin], 'float64')
            for index_z in xrange(self.nzmax_gg[zbin]):
                scalefac[index_z] = 1./(1.+self.z_gg[zbin][index_z])

            if ('bias_z{:}'.format(zbin + 1) in data.mcmc_parameters):
                bias = data.mcmc_parameters['bias_z{:}'.format(zbin + 1)]['current'] * data.mcmc_parameters['bias_z{:}'.format(zbin + 1)]['scale']

            #mask1 = self.z_gg[:, zbin] >= self.z_min_foreground[zbin]
            #mask2 = self.z_gg[:, zbin] <= self.z_max_foreground[zbin]
            #mask = mask1 & mask2
            interp_pz_foreground = itp.splev(self.z_gg[zbin], self.spline_pz_foreground[zbin])
            pr_foreground = interp_pz_foreground * dzdr

            ### Alessio
            #####################################
            ### Matter Power Spectrum ###########
            #####################################

            pk = self.get_matter_power_spectrum(r, self.z_gg[zbin], ells, self.nzmax_gg[zbin], self.nellsmax, cosmo, data)

            ### Alessio
            ######################################################################
            ### Actual calculation of theoretical galaxy- matter power spectrum  ###
            ######################################################################

            # Start loop over l for computation of C_l^shear
            Pl_GG_integrand = np.zeros(self.nzmax_gg[zbin], 'float64')
            screening_scale = pow(10.,data.mcmc_parameters['screening_scale']['current'])
            #screening_scale = data.mcmc_parameters['screening_scale']['current']

            dr = r[1:] - r[:-1]
            # FK: removed shifts like array[1:, ...]; only necessary if z[0]=0!
            for index_ell in xrange(self.nellsmax):
                mu = np.zeros(self.nzmax_gg[zbin], 'float64')
                #mu = np.zeros((self.nzmax_mm_out, self.nzmax_gm_out), 'float64')
                #kvec = np.zeros((self.nellsmax, self.nzmax_gm_out), 'float64')
                for index_z in xrange(self.nzmax_gg[zbin]):
                    # standard Limber approximation:
                    #k = ells[index_ells] / r[index_z]
                    # extended Limber approximation (cf. LoVerde & Afshordi 2008):
                    k = (ells[index_ell] + 0.5) / r[index_z]
                    a = (-k*k)*(2./3.)*scalefac[index_z]/poissonfactor
                    mu[index_z] = cosmo.get_poissonratio_screened(k, self.z_gg[zbin][index_z], screening_scale, a) #*(-k*k)*(2./3.)*scalefac[index_z]/poissonfactor
                    
                if self.use_quad_for_GG:
                    integral, err = integr.quad(lambda z: self.get_Pl_GG_integrand(ells[index_ell], z, zbin, cosmo, data), self.z_min_foreground[zbin], self.z_max_foreground[zbin])
                    # this is identical to quad:
                    #integral, err = logquad(lambda z: self.get_Pl_GG_integrand(ells[index_ell], z, zbin, cosmo, data), self.z_min_foreground[zbin], self.z_max_foreground[zbin])
                    Pl_GG[index_lin] = integral
                else:
                    Pl_GG_integrand[:] = pr_foreground[:] * pr_foreground[:] / r**2 * pk[index_ell, :] #* mu[:] * mu[:]   # Alessio: modified accordingly to modified weight above
                    Pl_GG[index_lin] = np.sum(0.5 * (Pl_GG_integrand[1:] + Pl_GG_integrand[:-1]) * dr)
                    #Pl_GG[index_ell, zbin] = integr.trapz(Pl_GG_integrand, r)
                    Pl_GG[index_lin] *= bias**2
                    Pl_GG[index_lin] *= ell_norm[index_ell]

                index_lin += 1

            #np.savetxt(self.path_save + 'P_GG_theory.txt', Pl_GG)
        return Pl_GG

    def get_Pl_GG_integrand(self, ell, z, zbin, cosmo, data):

        r, dzdr = cosmo.z_of_r(np.array([z]))

        pr = self.interp_nz_fg[zbin](z) * dzdr

        k = (ell + 0.5) / r
        norm = ell * ell / (2. * np.pi)
        integrand = norm * pr * pr / r**2 * self.get_matter_power_spectrum_for_GG(k, z, cosmo, data)

        return integrand



##########################################################################################################################################################
##########################################################################################################################################################


### Alessio
####################################
### Baryonic corrections  ##########
####################################



    def baryon_feedback_bias_sqr(self, k, z, A_bary=1.):
        """

        Fitting formula for baryon feedback following equation 10 and Table 2 from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)

        """

        # k is expected in h/Mpc and is divided in log by this unit...
        x = np.log10(k)

        a = 1. / (1. + z)
        a_sqr = a * a

        constant = {'AGN':   {'A2': -0.11900, 'B2':  0.1300, 'C2':  0.6000, 'D2':  0.002110, 'E2': -2.0600,
                              'A1':  0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1':  1.8400,
                              'A0':  0.15000, 'B0':  1.2200, 'C0':  1.3800, 'D0':  0.001300, 'E0':  3.5700},
                    'REF':   {'A2': -0.05880, 'B2': -0.2510, 'C2': -0.9340, 'D2': -0.004540, 'E2':  0.8580,
                              'A1':  0.07280, 'B1':  0.0381, 'C1':  1.0600, 'D1':  0.006520, 'E1': -1.7900,
                              'A0':  0.00972, 'B0':  1.1200, 'C0':  0.7500, 'D0': -0.000196, 'E0':  4.5400},
                    'DBLIM': {'A2': -0.29500, 'B2': -0.9890, 'C2': -0.0143, 'D2':  0.001990, 'E2': -0.8250,
                              'A1':  0.49000, 'B1':  0.6420, 'C1': -0.0594, 'D1': -0.002350, 'E1': -0.0611,
                              'A0': -0.01660, 'B0':  1.0500, 'C0':  1.3000, 'D0':  0.001200, 'E0':  4.4800}}

        A_z = constant[self.baryon_model]['A2']*a_sqr+constant[self.baryon_model]['A1']*a+constant[self.baryon_model]['A0']
        B_z = constant[self.baryon_model]['B2']*a_sqr+constant[self.baryon_model]['B1']*a+constant[self.baryon_model]['B0']
        C_z = constant[self.baryon_model]['C2']*a_sqr+constant[self.baryon_model]['C1']*a+constant[self.baryon_model]['C0']
        D_z = constant[self.baryon_model]['D2']*a_sqr+constant[self.baryon_model]['D1']*a+constant[self.baryon_model]['D0']
        E_z = constant[self.baryon_model]['E2']*a_sqr+constant[self.baryon_model]['E1']*a+constant[self.baryon_model]['E0']

        # only for debugging; tested and works!
        #print 'AGN: A2=-0.11900, B2= 0.1300, C2= 0.6000, D2= 0.002110, E2=-2.0600'
        #print self.baryon_model+': A2={:.5f}, B2={:.5f}, C2={:.5f}, D2={:.5f}, E2={:.5f}'.format(constant[self.baryon_model]['A2'], constant[self.baryon_model]['B2'], constant[self.baryon_model]['C2'],constant[self.baryon_model]['D2'], constant[self.baryon_model]['E2'])

        # original formula:
        #bias_sqr = 1.-A_z*np.exp((B_z-C_z)**3)+D_z*x*np.exp(E_z*x)
        # original formula with a free amplitude A_bary:
        bias_sqr = 1. - A_bary * (A_z * np.exp((B_z * x - C_z)**3) - D_z * x * np.exp(E_z * x))

        return bias_sqr

    def get_factor_IA(self, z, linear_growth_rate, rho_crit, small_h, Omega_m, amplitude, exponent):

        const = 5e-14 / small_h**2 # in Mpc^3 / M_sol

        # arbitrary convention
        z0 = 0.3
        #print utils.growth_factor(z, self.Omega_m)
        #print self.rho_crit
        factor = -1. * amplitude * const * rho_crit * Omega_m / linear_growth_rate * ((1. + z) / (1. + z0))**exponent

        return factor

    def get_critical_density(self, small_h):
        """
        The critical density of the Universe at redshift 0.

        Returns
        -------
        rho_crit in solar masses per cubic Megaparsec.

        """

        # yay, constants...
        Mpc_cm = 3.08568025e24 # cm
        M_sun_g = 1.98892e33 # g
        G_const_Mpc_Msun_s = M_sun_g * (6.673e-8) / Mpc_cm**3.
        H100_s = 100. / (Mpc_cm * 1.0e-5) # s^-1

        rho_crit_0 = 3. * (small_h * H100_s)**2. / (8. * np.pi * G_const_Mpc_Msun_s)

        return rho_crit_0
