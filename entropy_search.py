import logging
import math
import time
from random import sample

import mnist
import numpy as np
import scipy.stats as sts
from emcee import EnsembleSampler
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
from george.gp import GP
from george.kernels import Matern52Kernel
from sklearn.svm import SVC

import epmgp
from acquisitions import expected_improvement, information_gain
from horseshoe import Horseshoe

covariance_prior_mean, covariance_prior_sigma = 1, 0


def sample_hypers(X, y, K=20):
    """
        Samples from the hyperparameter space using MCMC

        #### Priors:
        - lambda = uniform [-10, 2]
        - covariance amplitude = lognormal(0, 1)
        - noise variance = horseshoe with length 0.1

    """

    cov = 1
    kernel = cov * Matern52Kernel(metric=[1., 1.], ndim=2, axes=[0, 1])
    hyper_distribution = GP(kernel=kernel, mean=np.mean(y))
    hyper_distribution.compute(X, yerr=0.05)

    def log_prob(theta):
        """
        Loglikelihood of the current distribution over hypers + prior
        """
        cov, lamb, noise = theta

        prior = 0

        # Check first if all parameters are inside some reasonable bounds.
        # This is mostly due to avoid silly choices that could be ammissible
        # but numerically useless.

        if not 0 < cov < 100:
            return -np.inf

        # Log likelihood for lambda is 0 when lambda is inside its bounds
        if not -9 < lamb < 2:
            return -np.inf

        if not -20 < noise < 20:
            return -np.inf

        # Log probability for lognormal distribution
        prior += sts.lognorm.logpdf(cov, covariance_prior_mean, covariance_prior_sigma)

        # Horseshoe
        prior += Horseshoe(scale=0.1).logpdf(noise)

        hyper_distribution.kernel.set_parameter_vector([cov, lamb, lamb])
        try:
            hyper_distribution.compute(X, yerr=np.sqrt(noise))
        except:
            return -np.inf

        return prior + hyper_distribution.log_likelihood(y.reshape(-1), quiet=True)

    nwalkers, ndim, iterations = K, kernel.ndim + 1, 500
    sampler = EnsembleSampler(nwalkers, ndim, log_prob)
    state = sampler.run_mcmc(np.random.rand(nwalkers, ndim), 100, rstate0=np.random.get_state())
    sampler.reset()
    sampler.run_mcmc(state, iterations)
    return sampler.chain[:, -1]


def compute_pmin(mu: np.array, sigma: np.array, derivatives=False):
    """
        Compute p_min using EPMGP from GPyOpt

        ### Parameters:
        - `mu`: mean vector
            - `np.array(N,)`
        - `sigma`: covariance matrix
            - `np.array(N, N)`
        - `derivatives`: enable derivatives
            - bool


        ### Returns:
        - `pmin`: log distribution
            - np.array(N, 1)
        - `dLogP_dMu`: derivative respect to mean
        - `dLogP_dSigma`: derivative respect to covariance
        - `dLogP_dMu_dMu`
    """
    return epmgp.joint_min(mu, sigma, with_derivatives=derivatives)


def entropy_search(dataset, bounds):
    n_hyper_samples = 20    # K parameter
    n_gen_samples = 50      # Z parameter
    n_innovations = 20      # P parameter

    logging.info("Sampling hypeparameters..")
    # K samples with mcmc over GP hyperparameters:
    # - lambda
    # - covariance amplitude for Matérn kernel
    # - noise variance
    hyperparameters = sample_hypers(dataset["X"], dataset["y"], K=n_hyper_samples)

    Omega = []
    p_min = []
    U = []
    models = []
    representers = []
    means = []
    covariances = []

    for hyper in hyperparameters:
        kernel_cov, lamb, noise = hyper
        kernel = kernel_cov * Matern52Kernel(metric=[1., 1.], ndim=2, axes=[0, 1])
        regressor = GP(kernel=kernel, mean=np.mean(dataset["y"]))
        regressor.compute(dataset["X"], yerr=noise)
        models.append(regressor)

        # sample Z point from M and get predictive mean + covariance
        X_samples = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_gen_samples, dataset["X"].shape[1])
        )

        representers.append(X_samples)
        mean, cov = regressor.predict(dataset["y"], X_samples, return_cov=True)
        means.append(mean)
        covariances.append(cov)
        # compute their Expected Improvement
        logging.debug("Computing EI...")
        exp_improvement = expected_improvement(mean, cov, regressor.sample(X_samples))
        U.append(exp_improvement)

        # Compute p_min using EPMGP
        logging.debug("Computing pMin")
        p_min.append(compute_pmin(mean, cov, derivatives=True))

        # generate P noise vectors from a Gaussian(0, I_Z)
        # Q: Why save'em in memory when they can be generated on the fly?
        # A: This way the noise is the same for all IG iterations
        innovations = []
        logging.debug("Generating innovations..")
        for _ in range(n_innovations):
            # Generate a gaussian noise vector
            innovations.append(np.random.normal(size=n_gen_samples).reshape(-1, 1))

        Omega.append(innovations)

    # maximize information gain => minimize -information_gain()
    # FIXME: find a better strategy for the initial guess/guesses.
    # Maybe random + last good configuration?
    logging.info("Ready to optimize Information Gain")
    return minimize(
        fun=lambda x: -information_gain(x, models, p_min, representers, U, Omega, dataset),
        x0=dataset["X"][np.argmax(dataset["y"])],
        method='L-BFGS-B',
        bounds=bounds,
        )


def obj_function(configuration, dataset=None):
    if dataset is None:
        dataset = load_mnist(1)
    c, gamma = configuration
    grid = GridSearchCV(SVC(kernel="rbf"), {'C': [10**c], 'gamma': [10**gamma]}, n_jobs=-1, verbose=3, cv=5)
    grid.fit(dataset["X"], dataset["y"])
    return grid.best_score_


def generate_prior(data):

    C_values = [10**(x) for x in [-10, -5, 0, 5, 10]]
    gamma_values = [10**(x) for x in [-10, -5, 0, 5, 10]]

    grid = GridSearchCV(SVC(kernel="rbf"), {'C': C_values, 'gamma': gamma_values},
                        n_jobs=-1, verbose=3, cv=3)
    grid.fit(data["X"], data["y"])

    x_values = np.log10(np.array([(params["C"], params["gamma"]) for params in grid.cv_results_["params"]]))
    y_values = np.array(grid.cv_results_["mean_test_score"])

    return x_values, y_values


def load_mnist(training_size):
    dataset = {}

    train_x, train_y = mnist.train_images(), mnist.train_labels()
    test_x, test_y = mnist.test_images(), mnist.test_labels()

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    if training_size > 1:
        indices = sample(range(train_x.shape[0]), math.floor(train_x.shape[0]/training_size))
        dataset["X"] = train_x[indices]
        dataset["y"] = train_y[indices]
    else:
        dataset["X"] = train_x
        dataset["y"] = train_y

    dataset["X_test"] = test_x
    dataset["y_test"] = test_y

    return dataset


def main():
    """
        Test Entropy Search with a simple "complex" function.
    """
    iterations = 10

    logging.info("Loading dataset...")
    data = load_mnist(128)
    logging.info("Dataset loaded.")

    best_x = None
    best_y = None

    # Bayesian optimization needs a prior: this can be
    # derived by some knowledge about the function
    # or generated/sampled by some fancy strategy
    dataset = {}
    dataset["X"], dataset["y"] = generate_prior(data)
    logging.info("Prior generated")

    best_index = np.argmax(dataset["y"])
    best_x = dataset["X"][best_index]
    best_y = dataset["y"][best_index]

    # Optimization loop can finally start. The stopping criteria is
    # based on a fixed number of iterations but could take in account
    # a "min improvement" policy

    for _ in range(iterations):

        # Find the next candidate
        result = entropy_search(dataset, bounds=[(-10, 10), (-10, 10)])

        logging.info(f"Evaluating function at {result.x}")
        function_time = time.time()
        # Evaluate the function
        y = obj_function(result.x)
        function_time = time.time() - function_time

        performance = (y / best_y - 1)*100
        logging.info(f"Function value: {y} ({('+' if performance > 0 else '')}{'%.5f' % performance} %), \
            {'%.5f' % function_time}s")

        # Save the results
        dataset["X"] = np.vstack([dataset["X"], result.x])
        dataset["y"] = np.append(dataset["y"], np.array([y]))

        # Save the best candidate so far
        best_index = np.argmax(dataset["y"])
        best_x = dataset["X"][best_index]
        best_y = dataset["y"][best_index]

    # Optimization loop has ended, print the results
    logging.info(f"Best score {best_y}")
    logging.info(f"with configuration: {str(best_x)}")


if __name__ == "__main__":
    logging.basicConfig(format='Entropy Search (%(process)s) - %(levelname)s - %(message)s', level=logging.INFO)
    main()
