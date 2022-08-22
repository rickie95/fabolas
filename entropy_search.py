from random import random
from statistics import correlation
from emcee import EnsembleSampler
import epmgp
import numpy as np
import scipy
import scipy.stats as sts
from scipy.optimize import minimize
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
        prior += sts.lognorm.logpdf(cov, covariance_prior_mean, covariance_prior_sigma)

        # Horseshoe 
        prior += tfp.Horseshoe(scale=0.1).log_prob(noise).numpy()

        # Compute log likelihood of the current distribution with proposed values

        prob = hyper_distribution.log_marginal_likelihood(theta=[cov, lamb, noise])

        return prior + prob

    nwalkers, ndim, iterations = 20, kernel.n_dims, 50
    sampler = EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(np.random.rand(nwalkers, ndim), iterations, progress=True)
    return sampler.chain[:, -1]

def expected_improvement(mean, covariance, y_values):
    y_max = y_values.max()

    u = (mean - y_max) / covariance
    ei = covariance * (u * sts.norm.cdf(u) + sts.norm.pdf(u))

    ei[covariance <= 0.] = 0.

    return ei

def compute_pmin(mu, sigma):
    """
        Compute p_min using EPMGP from GPyOpt

        ### Parameters:
        - `mu`: mean vector
            - `np.array(N,)`
        - `sigma`: covariance matrix
            - `np.array(N, N)`

    
        ### Returns:
        - `pmin`: distribution
            - np.array(N, 1)
    """
    return epmgp.joint_min(mu, sigma)

def compute_innovations(x, model, representer_points, variance, noise):
    """
        Returns possible innovations for point 'x'

        ### Parameters
        - `x`: point to be "innovated"
            - np.array(N,)
        - `model`: Gaussian Process
        - `representer_points`: representer points of `model`
        - `variance`: variance of `x` wrt `model`
        - `noise`: noise vector
            - np.array(N,)

        ### Returns
        - `d_mu`
        - `d_sigma`
    """
    var_noise = 0.5*10**(-3) # the noise is estimated as very little in comparison to the GP variance
    # Compute the correlation matrix and get the element corresponding to x
    _, correlation_x_r = model.predict(np.concatenate(x, representer_points), return_cov=True)[-1, :-1]
    d_mu = np.convolve(np.convolve(correlation_x_r, variance), np.dot(np.linalg.cholesky(variance + var_noise), noise))
    
    d_sigma = np.convolve(np.convolve(correlation_x_r, variance), np.transpose(correlation_x_r))

    return d_mu, d_sigma

def entropy_search(dataset):
    # K samples with mcmc over GP hyperparameters: lambda, covariance amplitude for Matérn kernel + noise variance
    hyperparameters = sample_hypers()
    n_gen_samples = 20
    P = 20

    Omega = []
    p_min = []
    U = []
    models = []
    representers = []

    for hyper in hyperparameters:
        cov, lamb, noise = hyper
        kernel = cov * Matern(length_scale=lamb) + WhiteKernel(noise_level=noise)
        regressor = GaussianProcessRegressor(kernel=kernel).fit(dataset["X"], dataset["y"])
        models.append(regressor)

        # sample Z point from M and get predictive mean + covariance
        X_samples = np.random.rand((n_gen_samples, dataset["X"].shape[0]))
        representers.append(X_samples)
        mean, cov = regressor.predict(X_samples, return_cov=True)

        # compute their Expected Improvement
        exp_improvement = expected_improvement(mean, cov, regressor.sample_y(X_samples))
        U.append[exp_improvement]

        # Compute p_min using EPMGP
        p_min.append(compute_pmin(mean, cov))
        
        # generate P noise vectors from a Gaussian(0, I_Z)
        # FIXME: noise vectors should not be deterministic
        # also why save'em in memory when they can be generated on the fly?
        innovations = []
        for _ in range(P):
            # Generate a gaussian noise vector
            innovations.append(sts.norm.ppf(np.linspace(0, 1, P + 2)[1:-1]))

        Omega.append(innovations)

    def information_gain(test_point):
        """
        Returns the information gain value for `test_point`
        """
        a = 0
        for i, model in enumerate(models):
            _, testpoint_var = model.predict(test_point, return_std=True)
            testpoint_var = testpoint_var**2

            for _ in range(P):
                d_mu, d_sigma = compute_innovations(test_point, model, representers[i], testpoint_var, Omega[i])

                # Compute pmin from the updated posterior
                q_min = compute_pmin(mean + d_mu, cov + d_sigma)

                d_entropy = - np.sum(q_min * np.log(q_min) + U[i]) + \
                    np.sum(p_min[i] * (np.log(p_min[i] + U[i])))

                a += 1/P * d_entropy
        return 1/len(models) * a

    # maximize information gain => minimize -information_gain()
    # FIXME: find a better strategy for the initial guess/guesses. 
    # Maybe random + last good configuration?
    return minimize(fun=lambda x: -information_gain(x), x0=[0., 0., 0.])

