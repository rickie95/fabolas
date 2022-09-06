from distutils.command.config import config
from random import random, sample
from statistics import correlation
from emcee import EnsembleSampler
import epmgp
import numpy as np
import scipy
import time
import scipy.stats as sts
from scipy.optimize import minimize
from tensorflow_probability import distributions as tfp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
from tensorflow.keras.datasets import mnist
import math
import logging

covariance_prior_mean, covariance_prior_sigma = 1, 0

def sample_hypers(X, y):
    """
        Samples from the hyperparameter space using MCMC

        #### Priors:
        - lambda = uniform [-10, 2]
        - covariance amplitude = lognormal(0, 1)
        - noise variance = horseshoe with length 0.1

    """

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
    sampler.run_mcmc(np.random.rand(nwalkers, ndim), iterations)
    return sampler.chain[:, -1]

def expected_improvement(mean, covariance, y_values):
    y_max = y_values.max()

    if covariance.all() == 0:
        logging.warning("Covariance is zero.")

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
            - np.array(1, D)
        - `model`: Gaussian Process
        - `representer_points`: representer points of `model`
            - np.array(P, D) where P is the number of rep points
        - `variance`: variance of `x` wrt `model`
            - np.array(1, 1)
        - `noise`: noise vector
            - np.array(1, N)

        ### Returns
        - `d_mu`
        - `d_sigma`
    """
    var_noise = np.array([[0.5*10**(-4)]]) # the noise is estimated as very little in comparison to the GP variance
    # Compute the correlation matrix and get the element corresponding to x
    _, correlation_x_r = model.predict(np.concatenate((x, representer_points)), return_cov=True)
    correlation_x_r = (correlation_x_r[-1, :-1]).reshape(-1, 1) # vector (n_rep , 1)
    corr_x_r_variance = np.dot(correlation_x_r, np.linalg.inv(variance))
    d_mu = np.dot(corr_x_r_variance, np.linalg.cholesky(variance + var_noise))
    d_sigma = corr_x_r_variance * correlation_x_r.T

    return d_mu, d_sigma

def entropy_search(dataset):
    
    logging.info("Sampling hypeparameters..")
    # K samples with mcmc over GP hyperparameters: lambda, covariance amplitude for Matérn kernel + noise variance
    hyperparameters = sample_hypers(dataset["X"], dataset["y"])
    n_gen_samples = 20
    P = 20

    Omega = []
    p_min = []
    U = []
    models = []
    representers = []
    means = []
    covariances = []

    for hyper in hyperparameters:
        cov, lamb, noise = np.e**(hyper) # Convert from log scale
        kernel = cov * Matern(length_scale=lamb) + WhiteKernel(noise_level=noise)
        regressor = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(dataset["X"], dataset["y"])
        models.append(regressor)

        # sample Z point from M and get predictive mean + covariance
        X_samples = np.random.rand(n_gen_samples, dataset["X"].shape[1])
        representers.append(X_samples)
        mean, cov = regressor.predict(X_samples, return_cov=True)
        means.append(mean)
        covariances.append(cov)
        # compute their Expected Improvement
        logging.debug("Computing EI...")
        exp_improvement = expected_improvement(mean, cov, regressor.sample_y(X_samples))
        U.append(exp_improvement)

        # Compute p_min using EPMGP
        logging.debug("Computing pMin")
        p_min.append(compute_pmin(mean, cov))
        
        # generate P noise vectors from a Gaussian(0, I_Z)
        # also why save'em in memory when they can be generated on the fly?
        innovations = []
        logging.debug("Generating innovations..")
        for _ in range(P):
            # Generate a gaussian noise vector
            innovations.append(np.array(sts.norm.ppf(np.linspace(0, 1, P + 2)[1:-1])))

        Omega.append(innovations)

    def information_gain(test_point):
        """
        Returns the information gain value for `test_point`

        """
        a = 0
        for i, model in enumerate(models):
            _, testpoint_var = model.predict(test_point.reshape(1, -1), return_cov=True)
            # vectorize
            for p in range(P):
                d_mu, d_sigma = compute_innovations(test_point.reshape(1, -1), model, representers[i], testpoint_var, Omega[i][p])

                # Compute pmin from the updated posterior
                q_min = compute_pmin(means[i] + d_mu.reshape(-1), covariances[i] + d_sigma)

                d_entropy = - np.sum(np.exp(q_min) * (q_min + U[i])) + \
                    np.sum(np.exp(p_min[i]) * (p_min[i] + U[i]))

                a += 1/P * d_entropy

        logging.info(f"IG: {1/len(models) * a} for test point: {test_point}")  
        return 1/len(models) * a

    # maximize information gain => minimize -information_gain()
    # FIXME: find a better strategy for the initial guess/guesses. 
    # Maybe random + last good configuration?
    logging.info("Ready to optimize Information Gain")
    return minimize(
        fun=lambda x: -information_gain(x), 
        x0=dataset["X"][np.argmax(dataset["y"])],
        method='L-BFGS-B',
        bounds=[(-10, 10), (-10, 10)],
        options={'maxiter': 10, 'maxfun': 10})

def obj_function(configuration, dataset):
    c, gamma = configuration
    classifier = SVC(C=math.e**c, gamma=math.e**gamma).fit(dataset["X"], dataset["y"])
    predictions = classifier.predict(dataset["X_test"])
    return zero_one_loss(dataset["y_test"], predictions)


def generate_prior(obj_function, dataset):

    C = np.array([-5, 0, 5])
    gamma = np.array([-5, 0, 5])

    x_values, y_values = [], []

    for c in C:
        for g in gamma:
            y = obj_function((c,g), dataset)
            x_values.append((c,g))
            y_values.append(y)

    return np.array(x_values), np.array(y_values)

def load_mnist(training_size):
    dataset = {}

    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    indices = sample(range(1, train_x.shape[0]), math.floor(train_x.shape[0]/training_size))

    dataset["X"] = train_x[indices]
    dataset["y"] = train_y[indices]

    dataset["X_test"] = test_x
    dataset["y_test"] = test_y

    return dataset

def main():
    """
        Test Entropy Search with a simple "complex" function.
    """
    iterations = 30

    data = load_mnist(128)

    best_x = None
    best_y = None

    # Bayesian optimization needs a prior: this can be
    # derived by some knowledge about the function
    # or generated/sampled by some fancy strategy
    dataset = {}
    dataset["X"], dataset["y"] = generate_prior(obj_function, data)
    logging.info("Prior generated")

    # Optimization loop can finally start. The stopping criteria is
    # based on a fixed number of iterations but could take in account
    # a "min improvement" policy 

    for _ in range(iterations):

        # Find the next candidate
        result = entropy_search(dataset)

        logging.info(f"Evaluating function at {result.x[0]}")
        function_time = time.time()
        # Evaluate the function
        y = obj_function(result.x[0], data)
        function_time = time.time() - function_time

        performance = "-" if best_y is not None else str((y / best_y - 1)*100 )
        logging.info(f"Function value: {y} ({performance}%), {function_time}s")

        # Save the results
        dataset["X"].append(candidate)
        dataset["y"].append(y)

        # Save the best candidate so far
        best_index = np.argmin(dataset["y"])
        best_x = dataset["X"][best_index]
        best_y = dataset["y"][best_index]
    

    # Optimization loop has ended, print the results
    logging.info(f"Best score {best_y}")
    logging.info(f"with configuration: {str(best_x)}")

if __name__ == "__main__":
    logging.basicConfig(format='%(process)s - %(levelname)s - %(message)s', level=logging.INFO)
    main()