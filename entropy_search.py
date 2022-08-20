import emcee
import numpy as np
import scipy
from tensorflow_probability import distributions as tfp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import math

covariance_prior_mean, covariance_prior_sigma = 1, 0

def sample_hypers(X, y):
    # Priors:
    # lambda = uniform [-10, 2]
    # covariance amplitude = lognormal(0, 1)
    # noise variance = horseshoe with length 0.1
    
    cov = 1
    kernel = cov * Matern() + WhiteKernel()
    hyper_distribution = GaussianProcessRegressor(kernel=kernel)
    hyper_distribution.fit(X, y)

    def log_prob(theta):
        """
        Loglikelihood of the current distribution over hypers + prior
        """
        cov, lamb, noise = theta

        prior = 0

        # Log likelihood for lambda is 0 when lambda is inside its bounds
        if not -10 < lamb < 2:
            return -np.inf

        # Log probability for lognormal distribution
        prior += scipy.stats.lognorm.logpdf(cov, covariance_prior_mean, covariance_prior_sigma)

        # Horseshoe 
        prior += tfp.Horseshoe(scale=0.1).log_prob(noise).numpy()

        # Compute log likelihood of the current distribution with proposed values

        prob = hyper_distribution.log_marginal_likelihood(theta=[cov, lamb, noise])

        return prior + prob

    nwalkers, ndim, iterations = 20, kernel.n_dims, 50
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(np.random.rand(nwalkers, ndim), iterations, progress=True)
    return sampler.chain[:, -1]

def expected_improvement(mean, covariance, y_values):
    y_max = y_values.max()

    u = (mean - y_max) / sigma
    ei = 0 # FIXME

    return ei

def entropy_search(dataset):
    # K samples with mcmc over GP hyperparameters: lambda, covariance amplitude for Matérn kernel + noise variance
    hyperparameters = sample_hypers()
    n_gen_samples = 20

    Omega = []
    for hyper in hyperparameters:
        cov, lamb, noise = theta
        kernel = cov * Matern(length_scale=lamb) + WhiteKernel(noise_level=noise)
        regressor = GaussianProcessRegressor(kernel=kernel).fit(X, y)
        # sample Z point from M and get predictive mean + covariance
        X_samples = np.random.rand((n_gen_samples, X.shape[0]))
        mean, cov = regressor.predict(X_samples, return_cov=True)
        # compute their Expected Improvement
        exp_improvement = expected_improvement(mean, cov, y_samples)

        p_min[i] = compute_pmin(mean, cov_matrix)
        
        # generate P noise vectors from a Gaussian(0, I_Z)
        innovations = []
        Omega.append(innovations)

    return information_gain()

