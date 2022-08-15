import emcee
import numpy as np
import scipy
import tensorflow_probability as tfp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def sample_hypers():
    # Priors:
    # lambda = uniform [-10, 2]
    # covariance amplitude = lognormal(0, 1)
    # noise variance = horseshoe with length 0.1
    nwalkers, ndim, iterations = 100, 3, 50
    kernel = Matern(length_scale=1) # FIXME
    hyper_distribution = GaussianProcessRegressor(kernel=kernel)

    def log_prob(theta):
        """
        Loglikelihood of the current distribution over hypers + prior
        """
        lamb, cov, noise = theta

        prior = 0

        # Log likelihood for lambda is 0 when lambda is inside its bounds
        if not -10 < lamb < 2:
            return -np.inf

        # Log probability for lognormal distribution
        prior += scipy.stats.lognorm.logpdf(cov, 0, loc=1)

        # Horseshoe 
        prior += tfp.distributions.Horseshoe(scale=0.1).log_prob(noise)

        # Compute log likelihood of the current distribution with proposed values
        prob = hyper_distribution.log_marginal_likelihood(theta=[lamb, cov])
        return prior

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(np.random.rand(nwalkers, ndim), iterations)
    sampler.chain[:, -1]


def entropy_search(dataset):
    # K samples with mcmc over GP hyperparameters: lambda, covariance amplitude for Matérn kernel + noise variance
    hyperparameters = sample_hypers()

    Omega = []
    for hyper in hyperparameters:
        # fit GP Regressor M with hyper
        # sample Z point from M
        # compute their Expected Improvement
        # get mean and cov_matrix from M
        p_min[i] = compute_pmin(mean, cov_matrix)
        
        # generate P noise vectors from a Gaussian(0, I_Z)
        innovations = [ ]
        Omega.append(innovations)

    return information_gain()

if __name__ == "__main__":
    sample_hypers()