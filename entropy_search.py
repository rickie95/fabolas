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
    _, correlation_x_r = model.predict(np.concatenate((x, representer_points)), return_cov=True)
    correlation_x_r = correlation_x_r[-1, :-1]
    d_mu = np.convolve(np.convolve(correlation_x_r, variance), np.dot(np.linalg.cholesky(variance + var_noise), noise))
    # FIXME Cholesky decomposition must be computed on a matrix NxN, but `variance` is a vector
    d_sigma = np.convolve(np.convolve(correlation_x_r, variance), np.transpose(correlation_x_r))

    return d_mu, d_sigma

def entropy_search(dataset):
    
    print("Sampling hypeparameters..")
    # K samples with mcmc over GP hyperparameters: lambda, covariance amplitude for Matérn kernel + noise variance
    hyperparameters = sample_hypers(dataset["X"], dataset["y"])
    n_gen_samples = 50
    P = 20

    Omega = []
    p_min = []
    U = []
    models = []
    representers = []

    for hyper in hyperparameters:
        cov, lamb, noise = np.e**(hyper) # Convert from log scale
        kernel = cov * Matern(length_scale=lamb) + WhiteKernel(noise_level=noise)
        regressor = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(dataset["X"], dataset["y"])
        models.append(regressor)

        # sample Z point from M and get predictive mean + covariance
        X_samples = np.random.rand(n_gen_samples, dataset["X"].shape[1])
        representers.append(X_samples)
        mean, cov = regressor.predict(X_samples, return_cov=True)

        # compute their Expected Improvement
        print("Computing EI...")
        exp_improvement = expected_improvement(mean, cov, regressor.sample_y(X_samples))
        U.append(exp_improvement)

        # Compute p_min using EPMGP
        print("Computing pMin")
        p_min.append(compute_pmin(mean, cov))
        
        # generate P noise vectors from a Gaussian(0, I_Z)
        # FIXME: noise vectors should not be deterministic
        # also why save'em in memory when they can be generated on the fly?
        innovations = []
        print("Generating innovations..")
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
            _, testpoint_var = model.predict(test_point.reshape(1, -1), return_cov=True)

            for _ in range(P):
                d_mu, d_sigma = compute_innovations(test_point.reshape(1, -1), model, representers[i], testpoint_var, Omega[i])

                # Compute pmin from the updated posterior
                q_min = compute_pmin(mean + d_mu, cov + d_sigma)

                d_entropy = - np.sum(q_min * np.log(q_min) + U[i]) + \
                    np.sum(p_min[i] * (np.log(p_min[i] + U[i])))

                a += 1/P * d_entropy

        print(f"IG: {1/len(models) * a}")  
        return 1/len(models) * a

    # maximize information gain => minimize -information_gain()
    # FIXME: find a better strategy for the initial guess/guesses. 
    # Maybe random + last good configuration?
    print("Ready to optimize Information Gain")
    return minimize(fun=lambda x: -information_gain(x), x0=dataset["X"][np.argmax(dataset["y"])])

def obj_function(configuration, dataset):
    c, gamma = configuration
    classifier = SVC(C=c, gamma=gamma).fit(dataset["X"], dataset["y"])
    predictions = classifier.predict(dataset["X_test"])
    return zero_one_loss(dataset["y_test"], predictions)


def generate_prior(obj_function, dataset):

    C = np.array([1, 10, 100])
    gamma = np.array([10**(-3), 10**(-5), 10**(-7)])

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
    print("Prior generated")

    # Optimization loop can finally start. The stopping criteria is
    # based on a fixed number of iterations but could take in account
    # a "min improvement" policy 

    for _ in range(iterations):

        # Find the next candidate
        candidate = entropy_search(dataset)

        print("Evaluating function..")
        function_time = time.time()
        # Evaluate the function
        y = obj_function(candidate, data)
        function_time = time.time() - function_time

        performance = "-" if best_y is not None else str((y / best_y - 1)*100 )
        print(f"Function value: {y} ({performance}%), {function_time}s")

        # Save the results
        dataset["X"].append(candidate)
        dataset["y"].append(y)

        # Save the best candidate so far
        best_index = np.argmin(dataset["y"])
        best_x = dataset["X"][best_index]
        best_y = dataset["y"][best_index]
    

    # Optimization loop has ended, print the results
    print(f"Best score {best_y}")
    print(f"with configuration: {str(best_x)}")

if __name__ == "__main__":
    main()