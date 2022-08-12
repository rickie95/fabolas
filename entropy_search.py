import emcee
import numpy as np

def sample_hypers():
    # Priors:
    # lambda = uniform [-10, 2]
    # covariance amplitude = lognormal(0, 1)
    # noise variance = horseshoe with length 0.1


    def log_prior(theta):
        lamb, cov, noise = theta
        if -10 < lamb < 2 and True: # FIXME: see EMCEE docs
            return 0.0
        return -np.inf

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
