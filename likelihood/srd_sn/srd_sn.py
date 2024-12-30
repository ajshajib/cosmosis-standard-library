"""
The likelihood module for the Pantheon dataset discards
much of the JLA/CosmoMC SN machinery, which is not needed
for Pantheon because the systematic errors in that dataset
have been subsumed into a single systematic covariance matrix.

In consequence almost all the terms in those codes are zero.

"""

from cosmosis.gaussian_likelihood import GaussianLikelihood
from cosmosis.datablock import names
import os
import numpy as np


# Default is to use the binned version of the data since it's much faster
# You can also downloaded and run the full data if you like, and set the data_file
# and covmat_file parameters in the ini file.
default_data_file = os.path.join(
    os.path.split(__file__)[0], "lcparam_Y10_DDF_WFD_3.0xFOUNDATION_noScatter.txt"
)
default_covmat_file = os.path.join(
    os.path.split(__file__)[0], "sys_Y10_DDF_WFD_FOUNDATION_2.txt"
)


class SRDSNLikelihood(GaussianLikelihood):
    x_section = names.distances
    x_name = "z"
    y_section = names.distances
    y_name = "D_A"
    like_name = "srdsn"

    def build_data(self):
        """
        Run once at the start to load in the data vectors.

        Returns x, y where x is the independent variable (redshift in this case)
        and y is the Gaussian-distribured measured variable (magnitude in this case).

        """
        filename = self.options.get_string("data_file", default=default_data_file)
        print("Loading Pantheon data from {}".format(filename))

        # The Pantheon data format is mostly zeros.
        # The only columns that we actually need here are the redshift,
        # magnitude, and magnitude error.
        data = np.genfromtxt(filename).T
        z = data[1]
        m_obs = data[4]

        self.z_cmb = data[1]
        self.z_hel = data[2]
        # We will need mag_obs_err later, when building the covariance,
        # so save it for now.
        self.mag_obs_err = data[5]

        # Return this to the parent class, which will use it
        # when working out the likelihood
        print(
            "Found {} Pantheon supernovae (or bins if you used the binned data file)".format(
                len(z)
            )
        )
        return z, m_obs

    def build_covariance(self):
        """Run once at the start to build the covariance matrix for the data"""
        filename = self.options.get_string("covmat_file", default=default_covmat_file)
        print("Loading Pantheon covariance from {}".format(filename))
        # The file format for the covariance has the first line as an integer
        # indicating the number of covariance elements, and the the subsequent
        # lines being the elements.
        # This data file is just the systematic component of the covariance -
        # we also need to add in the statistical error on the magnitudes
        # that we loaded earlier
        f = open(filename)
        line = f.readline()
        n = int(line)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = float(f.readline())

        # Now add in the statistical error to the diagonal
        for i in range(n):
            C[i, i] += self.mag_obs_err[i] ** 2
        f.close()

        C *= 1 / 0.67  # calibrateable systematics

        # Return the covariance; the parent class knows to invert this
        # later to get the precision matrix that we need for the likelihood.
        return C

    def extract_theory_points(self, block):
        """
        Run once per parameter set to extract the mean vector that our
        data points are compared to.  In this case that means the theory
        prediction for the magnitudes.

        Note that because I'm not a supernova person I will use the absolute
        magnitude to change the *theory*, not to change the *data*.  This
        doesn't make any difference to the mathematics, but does maintain the
        conceptual distinction between "theory" and "observation" better
        than the usual way around.

        """
        # import scipy.interpolate

        # Pull out mu and z from the block.
        # self.x_section etc. are defined above - we make them variables
        # so that the user can override them in the ini file.
        # We have to cut off the first element z=0, because mu is not finite
        # there and this confuses the interpolator.
        # theory_x = block[self.x_section, self.x_name][1:]
        # theory_y = block[self.y_section, self.y_name][1:]

        # # This makes an interpolation function
        # f = scipy.interpolate.interp1d(theory_x, theory_y, kind=self.kind)

        # # Actually do the interpolation at the data redshifts
        # theory = np.atleast_1d(f(self.data_x))

        # # Add the absolute supernova magnitude and return
        # M = block[names.supernova_params, "m"]
        # return theory + M

        import scipy.interpolate

        # Pull out theory DA and z from the block.
        theory_x = block[self.x_section, self.x_name]
        theory_y = block[self.y_section, self.y_name]
        theory_ynew = np.zeros_like(self.z_cmb)

        # Interpolation function of theory so we can evaluate at redshifts of the data
        f = scipy.interpolate.interp1d(theory_x, theory_y, kind=self.kind)

        # distance modulus
        theory_ynew = (
            5.0
            * np.log10(
                (1 + self.z_cmb) * (1 + self.z_hel) * np.atleast_1d(f(self.z_cmb))
            )
            + 25.0
        )

        # This offset M will be marginalized in the modified log likelihood computation
        M = block[names.supernova_params, "M"]
        return theory_ynew + M

    def do_likelihood(self, block):
        # get data x by interpolation
        x = np.atleast_1d(self.extract_theory_points(block))
        mu = np.atleast_1d(self.data_y)

        # If covariance is a function of parameters, compute the
        # new one now.
        if not self.constant_covariance:
            self.cov = np.atleast_2d(self.extract_covariance(block))
            self.inv_cov = np.atleast_2d(self.extract_inverse_covariance(block))

        # #gaussian likelihood
        # d = x-mu
        # chi2 = np.einsum('i,ij,j', d, self.inv_cov, d)
        # chi2 = float(chi2)
        # like = -0.5*chi2

        # start modified log-likelihood computation to marginalize offset M
        like = cov_log_likelihood(x, mu, self.inv_cov)
        chi2 = -2.0 * like

        # It can be useful to save the chi^2 as well as the likelihood,
        # especially when the covariance is non-constant.
        block[names.data_vector, self.like_name + "_CHI2"] = chi2
        block[names.data_vector, self.like_name + "_N"] = mu.size

        # if the covariance is a function of parameters then we must
        # account for this in the likelihood.
        if not self.constant_covariance:
            log_det = self.extract_covariance_log_determinant(block)
        else:
            log_det = self.log_det_constant

        norm = -0.5 * log_det
        like += norm
        block[names.data_vector, self.like_name + "_LOG_DET"] = float(log_det)
        block[names.data_vector, self.like_name + "_NORM"] = float(norm)

        # Numpy has started returning a 0D array in recent versions (1.14).
        # Convert this to a float.
        like = float(like)

        # Now save the resulting likelihood
        block[names.likelihoods, self.like_name + "_LIKE"] = like

        # For some very fast likelihoods the overhead from
        # the steps below is painful.  Setting likelihood_only avoids that.
        if self.likelihood_only:
            return

        # And also the predicted data points - the vector of observables
        # that in a fisher approch we want the derivatives of.
        # and inverse cov mat which also goes into the fisher matrix.
        block[names.data_vector, self.like_name + "_theory"] = x
        block[names.data_vector, self.like_name + "_data"] = mu
        block[names.data_vector, self.like_name + "_inverse_covariance"] = self.inv_cov

        # We might just be calculating the inverse cov and ignoring the covmat.
        # in that case we do not try to save it
        if self.cov is not None:
            block[names.data_vector, self.like_name + "_covariance"] = self.cov
            # Also save a simulation of the data - the mean with added noise
            # these can be used among other places by the ABC sampler.
            # This also requires the cov mat.
            # If we have a parameter-dependent covariance we need
            # to re-calculate the Cholesky decomposition to simulate some data.
            if not self.constant_covariance:
                self.chol = np.linalg.cholesky(self.cov)
            sim = self.simulate_data_vector(x)
            block[names.data_vector, self.like_name + "_simulation"] = sim


def cov_log_likelihood(mu_model, mu, inv_cov):
    """
    Computes modified likelihood computation to marginalize offset M
    from https://arxiv.org/abs/astro-ph/0104009v1, see Equation A9-12
    """
    delta = np.array([mu_model - mu])
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov @ deltaT)
    B = np.sum(delta @ inv_cov)
    C = np.sum(inv_cov)
    chi2 = chit2 - (B**2 / C) + np.log(C / (2 * np.pi))
    return -0.5 * chi2


# This takes our class and turns it into
setup, execute, cleanup = SRDSNLikelihood.build_module()
