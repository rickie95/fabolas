import logging
import time

import numpy as np
import scipy.stats as sts
from emcee.ensemble import EnsembleSampler
from george.gp import GP
from george.kernels import ConstantKernel, LinearKernel, Matern52Kernel
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

import epmgp
from acquisitions import (expected_improvement, information_gain,
                          predict_testpoint_george)
from datasets import load_mnist
from horseshoe import Horseshoe

min_time = 0

def information_gain_cost(test_point, cost_models, models, dataset, p_min, representers, U, Omega):
    overhead_cost = 0.0001
    predicted_cost, _ = predict_testpoint_george(cost_models, dataset["c"], test_point)
    cost_factor = 1/(predicted_cost + overhead_cost)
    ig = information_gain(test_point, models, p_min, representers, U, Omega, dataset, enable_log=False)
    ig_cost = cost_factor * ig
    logging.info(f"IG: {ig}, cost_f {cost_factor}, ig_cost {ig_cost}. x = {test_point}")
    return ig_cost


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

    nwalkers, iterations = K, 500

    sampler = EnsembleSampler(
        nwalkers=20,
        ndim=len(kernel) - 1,
        log_prob_fn=log_likelihood,
        args=[function_regressor, X, y])

    state = sampler.run_mcmc(np.random.rand(nwalkers, len(kernel) - 1), 100, rstate0=np.random.get_state())
    sampler.reset()
    sampler.run_mcmc(state, iterations)

    return sampler.chain[:, -1]


def fabolas(dataset, bounds):
    n_hyper_samples = 20    # K parameter
    n_gen_samples = 50      # Z parameter
    n_innovations = 20      # P parameter

    X = dataset["X"]

    logging.info("Sampling hyperparameters for function models...")
    hypers = sample_hypers(
        X=np.concatenate((X[:, :-1], ((1 - X[:, -1])**2).reshape(-1, 1)), axis=1),
        y=dataset["y"],
        K=n_hyper_samples)

    logging.info("Sampling hyperparameters for cost models...")
    cost_hypers = sample_hypers(X, dataset["c"], K=n_hyper_samples)

    models = []
    cost_models = []
    representers = []
    means = []
    covariances = []
    U = []
    Omega = []
    p_min = []

    for h in hypers:
        cov, lamb1, lamb2, noise = h
        kernel = cov * Matern52Kernel(
            metric=np.exp([lamb1, lamb1]),
            ndim=3,
            axes=[0, 1]
            ) * (
                LinearKernel(log_gamma2=lamb2, order=1, ndim=3, axes=[2]) +
                ConstantKernel(log_constant=lamb2, ndim=3, axes=[2])
                )

        regressor = GP(kernel=kernel, mean=np.mean(dataset["y"]))

        x = np.concatenate((X[:, :-1], ((1 - X[:, -1])**2).reshape(-1, 1)), axis=1)

        regressor.compute(x, np.sqrt(noise))

        models.append(regressor)

        X_samples = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_gen_samples, dataset["X"].shape[1])
        )

        representers.append(X_samples)
        mean, cov = regressor.predict(dataset["y"].reshape(-1), X_samples, return_cov=True)

        means.append(mean)
        covariances.append(cov)

        logging.debug("Computing EI...")
        exp_improvement = expected_improvement(mean, cov, regressor.sample(X_samples))
        U.append(exp_improvement)

        logging.debug("Computing pMin")
        p_min.append(epmgp.joint_min(mean, cov, with_derivatives=True))

        # generate P noise vectors from a Gaussian(0, I_Z)
        # Q: Why save'em in memory when they can be generated on the fly?
        # A: This way the noise is the same for all IG iterations
        innovations = []
        logging.debug("Generating innovations..")
        for _ in range(n_innovations):
            # Generate a gaussian noise vector
            innovations.append(np.random.normal(size=n_gen_samples).reshape(-1, 1))

        Omega.append(innovations)

    for ch in cost_hypers:
        cov, lamb1, lamb2, noise = h
        kernel = cov * Matern52Kernel(
            metric=np.exp([lamb1, lamb1]),
            ndim=3,
            axes=[0, 1]
            ) * (
                LinearKernel(log_gamma2=lamb2, order=1, ndim=3, axes=[2]) +
                ConstantKernel(log_constant=lamb2, ndim=3, axes=[2])
                )

        cost_regressor = GP(kernel=kernel, mean=np.mean(dataset["c"]))
        cost_regressor.compute(dataset["X"], np.sqrt(noise))
        cost_models.append(cost_regressor)

    logging.info("Ready to optimize Information Gain by Cost")
    return minimize(
        fun=lambda x: - information_gain_cost(
            x, cost_models, models, dataset, p_min, representers, U, Omega),
        x0=dataset["X"][np.argmin(dataset["y"])],
        method='L-BFGS-B',
        bounds=bounds,
        )


def obj_function(configuration, dataset, size=None):

    c, gamma = configuration

    grid = GridSearchCV(SVC(kernel="rbf"), {'C': [c], 'gamma': [gamma]}, n_jobs=-1, cv=5)
    grid.fit(dataset["X"], dataset["y"])
    executionTime = (grid.cv_results_["mean_fit_time"] + grid.cv_results_["mean_score_time"])[0]

    mean_score = grid.cv_results_['mean_test_score'][0]
    logging.info(f"Config: [C: {'%.3f' % c}, gamma: {'%.3f' % gamma}, \
        size: {dataset['size'] if size is None else size}] \
        Mean score: {'%.5f' % mean_score} \
        (Best {'%.5f' % grid.best_score_}) Time: {'%.3fs' % executionTime}")

    return 1 - grid.best_score_, executionTime


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
        data = load_mnist(1/size)
        data["size"] = 1/size
        for gamma in gamma_values:
            for c in C_values:
                score, cost = obj_function((c, gamma), data)
                dataset["X"].append((c, gamma, 1/size))
                dataset["y"].append(score)
                dataset["c"].append(cost)

    # Dump the prior into a csv
    prior = np.array([(*x, y, z) for x, y, z in zip(dataset["X"], dataset["y"], dataset["c"])])
    np.savetxt("./results/fabolas/prior.csv", prior, delimiter=",")

    # After the dump do some transformations on the data

    # Cost is in log form
    dataset["c"] = np.log10(dataset["c"])

    if min(dataset["c"]) < 0:
        dataset["min_time"] = min(dataset["c"])
        dataset["c"] += -dataset["min_time"]

    return dataset


def load_prior():
    data = np.loadtxt("./results/fabolas/partial-prior.csv", delimiter=",")

    dataset = {}
    dataset["X"] = data[:, :-2]
    
    dataset["y"] = np.array(data[:, -2]).reshape(-1, 1)
    dataset["c"] = np.array(data[:, -1]).reshape(-1, 1)

    # Configuration is mapped in log space
    dataset["X"][:, :-1] = np.log10(dataset["X"][:, :-1])

    # Cost is in log form
    dataset["c"] = np.log10(dataset["c"])

    dataset["min_time"] = 0
    if min(dataset["c"]) < 0:
        dataset["min_time"] = np.array(min(dataset["c"]))
        dataset["c"] += -dataset["min_time"]

    #  Validation error, the lesser the better
    dataset["y"] = 1 - dataset["y"]

    return dataset


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

    best_index = np.argmin(dataset["y"])
    best_x = dataset["X"][best_index]
    best_y = dataset["y"][best_index]

    # Optimization loop can finally start. The stopping criteria is
    # based on a fixed number of iterations but could take in account
    # a "min improvement" policy

    for i in range(iterations):
        logging.info("==================================")
        logging.info(f"========= Iteration # {i} =========")
        logging.info("==================================")
        # Find the next candidate
        result = fabolas(dataset, bounds=[(-10, 10), (-10, 10), (1/256, 1)])

        logging.info(f"Evaluating function at {result.x}")
        function_time = time.time()
        # Evaluate the function. Last component of X is the training set size
        y, cost = obj_function(10**result.x[:-1], load_mnist(result.x[-1]), result.x[-1])
        function_time = time.time() - function_time

        performance = (y / best_y - 1)*100
        logging.info(f"Function value: {y} ({('+' if performance > 0 else '')}{'%.5f' % performance} %), \
            {'%.5f' % cost}s")

        # Save the results
        dataset["X"] = np.vstack([dataset["X"], result.x])
        dataset["y"] = np.append(dataset["y"], np.array([y]))
        dataset["c"] = np.append(dataset["c"], np.array([np.log10(cost) - dataset["min_time"]]))

        if min(dataset["c"]) < 0:
            dataset["c"] += -(np.log10(cost) - dataset["min_time"])
            dataset["min_time"] = np.array(np.log10(cost))

        # FIXME: write the results in the CSV

        # Save the best candidate so far
        best_index = np.argmin(dataset["y"])
        best_x = dataset["X"][best_index]
        best_y = dataset["y"][best_index]

    # Optimization loop has ended, print the results
    logging.info(f"Best score {best_y}")
    logging.info(f"with configuration: {str(best_x)}")


if __name__ == "__main__":
    logging.basicConfig(
        format='FABOLAS (%(process)s) - %(levelname)s - %(message)s',
        level=logging.INFO)
    main()
