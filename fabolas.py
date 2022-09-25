import logging
import math
import random
import time

import numpy as np
import scipy.stats as sts
from emcee import EnsembleSampler
from george.gp import GP
from george.kernels import ConstantKernel, LinearKernel, Matern52Kernel
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import epmgp
from acquisitions import expected_improvement, information_gain_cost
from ard import AutomaticRelevanceDetermination
from datasets import load_mnist
from horseshoe import Horseshoe


covariance_prior_mean, covariance_prior_sigma = 1, 0


def log_likelihood(params, gp, X, y):
    """
        Log likelihood function expressing prior for GP hyperparameters

        params: (4,) float
            - covariance amplitude
            - lengthscale for Matérn5/2
            - lengthscale for Linear + Constant
            - noise
    """

    prior = 0
    cov, lamb1, lamb2, noise = params

    # Check first if all parameters are inside some reasonable bounds.
    # This is mostly due to avoid silly choices that could be ammissible
    # but numerically useless.

    if not 0 < cov < 20:
        return -np.inf

    # All lengthscales use an uniform prior
    if not -10 < lamb1 < 2:
        return -np.inf

    if not -9 < lamb2 < 2:
        return -np.inf

    if not -20 < noise < 20:
        return -np.inf

    # Log probability for lognormal distribution
    prior += sts.lognorm.logpdf(cov, 1, 0)

    # Horseshoe
    prior += Horseshoe(scale=0.1).logpdf(noise)

    # If parameters are reasonable, proceed to compute the marginal loglikelihood
    # from the GP

    # Update the kernel and compute the lnlikelihood.
    # [covariance, matern_length_dim_1, matern_length_dim2, linear_length, bias_length]
    gp.kernel.set_parameter_vector([cov, lamb1, lamb1, lamb2, lamb2])
    try:
        gp.compute(X, yerr=np.sqrt(noise))
    except:
        return -np.inf

    return prior + gp.log_likelihood(y.reshape(-1), quiet=True)


def sample_hypers(X, y, K=20):
    """
        Samples from the hyperparameter space using MCMC

        #### Priors:
        - lambda = uniform [-10, 2]
        - covariance amplitude = lognormal(0, 1)
        - noise variance = horseshoe with length 0.1

    """
    cov = 1
    # Kernel params:
    kernel = cov * Matern52Kernel(metric=np.array([0.1, 0.1]), ndim=3, axes=[0, 1]) * (
        LinearKernel(log_gamma2=np.log(1), order=1, ndim=3, axes=[2])
        + ConstantKernel(log_constant=np.log(1), ndim=3, axes=[2]))

    function_regressor = GP(kernel=kernel, mean=np.mean(y))
    x = np.concatenate((X[:, :-1], ((1 - X[:, -1])**2).reshape(-1, 1)), axis=1)

    nwalkers, iterations = K, 500

    sampler = EnsembleSampler(
        nwalkers=20,
        ndim=len(kernel) - 1,
        log_prob_fn=log_likelihood,
        args=[function_regressor, x, y])

    state = sampler.run_mcmc(np.random.rand(nwalkers, len(kernel) - 1), 100, rstate0=np.random.get_state())
    sampler.reset()
    sampler.run_mcmc(state, iterations)

    return sampler.chain[:, -1]


def obj_function(configuration, dataset):

    c, gamma = configuration

    startTime = time.time()
    grid = GridSearchCV(SVC(kernel="rbf"), {'C': [10**c], 'gamma': [10**gamma]}, n_jobs=-1, cv=5)
    grid.fit(dataset["X"], dataset["y"])
    executionTime = time.time() - startTime

    logging.info(f"Config: [C: {c}, gamma: {gamma}, size: {dataset['size']}] \t \
        Score: {grid.best_score_} \t Time: {executionTime}")

    return grid.best_score_, executionTime


def sample_from_training_set(data, size):
    n_examples = data["X"].shape[0]
    return random.sample(range(n_examples), math.floor(n_examples/size))


def generate_prior():
    C_values = [10**(x) for x in [-10, -5, 0, 5, 10]]
    gamma_values = [10**(x) for x in [-10, -5, 0, 5, 10]]
    s_values = [128, 64, 32, 4]

    # Init prior informations
    dataset = {}
    dataset["X"] = []
    dataset["y"] = []
    dataset["c"] = []

    for size in s_values:
        data = load_mnist(size)
        data["size"] = size
        for gamma in gamma_values:
            for c in C_values:
                score, cost = obj_function(np.log10((c, gamma)), data)
                dataset["X"].append((c, gamma, size))
                dataset["y"].append(score)
                dataset["c"].append(cost)

    # Dump the prior into a csv
    prior = np.array([(*x, y, z) for x, y, z in zip(dataset["X"], dataset["y"], dataset["c"])])
    np.savetxt("./results/fabolas/prior.csv", prior, delimiter=",")

    return dataset


def load_prior():
    data = np.loadtxt("./results/fabolas/prior.csv", delimiter=",")

    dataset = {}
    dataset["X"] = data[:, :-2]
    dataset["y"] = data[:, -2]
    dataset["c"] = data[:, -1]

    return dataset


def fabolas(dataset, bounds):
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
        kernel = kernel_cov * AutomaticRelevanceDetermination(length_scale=math.e**lamb, nu=5/2) + \
            WhiteKernel(noise_level=abs(noise))
        regressor = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(dataset["X"], dataset["y"])
        models.append(regressor)

        # sample Z point from M and get predictive mean + covariance
        X_samples = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_gen_samples, dataset["X"].shape[1])
        )

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
        p_min.append(epmgp.joint_min(mean, cov, derivatives=True))

        # generate P noise vectors from a Gaussian(0, I_Z)
        # Q: Why save'em in memory when they can be generated on the fly?
        # A: This way the noise is the same for all IG iterations
        innovations = []
        logging.debug("Generating innovations..")
        for _ in range(n_innovations):
            # Generate a gaussian noise vector
            innovations.append(np.random.normal(size=n_gen_samples))

        Omega.append(innovations)

    # maximize information gain => minimize -information_gain()
    # FIXME: find a better strategy for the initial guess/guesses.
    # Maybe random + last good configuration?
    logging.info("Ready to optimize Information Gain")
    return minimize(
        fun=information_gain_cost,
        args=(cost_model, models, p_min, representers, U, Omega),
        x0=dataset["X"][np.argmax(dataset["y"])],
        method='L-BFGS-B',
        bounds=bounds,
        )


def main():
    """
        Test FABOLAS with a simple "complex" function.
    """
    iterations = 10

    best_x = None
    best_y = None

    # Bayesian optimization needs a prior: this can be
    # derived by some knowledge about the function
    # or generated/sampled by some fancy strategy
    # dataset = generate_prior()
    dataset = load_prior()
    logging.info("Prior generated")

    best_index = np.argmax(dataset["y"])
    best_x = dataset["X"][best_index]
    best_y = dataset["y"][best_index]

    # Optimization loop can finally start. The stopping criteria is
    # based on a fixed number of iterations but could take in account
    # a "min improvement" policy

    for _ in range(iterations):

        # Find the next candidate
        result = fabolas(dataset, bounds=[(-10, 10), (-10, 10), (1, 1000)])
        # FIXME: upper bound for size should be the number of training examples

        logging.info(f"Evaluating function at {result.x}")
        function_time = time.time()
        # Evaluate the function. Last component of X is the training set size
        y, cost = obj_function(result.x[:-1], load_mnist(result.x[-1]))
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
    logging.basicConfig(
        format='FABOLAS (pid: %(process)s) - %(levelname)s - %(message)s',
        level=logging.INFO)
    main()
