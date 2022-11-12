import logging
import time

import numpy as np
import scipy.stats as sts
from emcee.ensemble import EnsembleSampler
from george.gp import GP
from george.kernels import ConstantKernel, LinearKernel, Matern52Kernel
from scipy.optimize import minimize

import epmgp
from acquisitions import (expected_improvement, information_gain,
                          predict_testpoint_george)
from horseshoe import Horseshoe


def information_gain_cost(test_point, cost_models, models, dataset, p_min, representers, U, Omega):
    overhead_cost = 0.0001
    predicted_cost, _ = predict_testpoint_george(
        cost_models, dataset["c"], test_point)
    cost_factor = 1/(predicted_cost + overhead_cost)
    ig = information_gain(test_point, models, p_min,
                          representers, U, Omega, dataset, enable_log=False)
    ig_cost = cost_factor * ig
    logging.info(
        f"IG: {ig}, cost_f {cost_factor}, ig_cost {ig_cost}. x = {test_point}")
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
        if noise < 0:
            return -np.inf

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

    state = sampler.run_mcmc(np.random.rand(nwalkers, len(
        kernel) - 1), 100, rstate0=np.random.get_state())
    sampler.reset()
    sampler.run_mcmc(state, iterations)

    return sampler.chain[:, -1]


def get_candidate(dataset, bounds):
    n_hyper_samples = 20    # K parameter
    n_gen_samples = 50      # Z parameter
    n_innovations = 20      # P parameter

    if len(dataset["size"].shape) == 1:
        dataset["size"] = dataset["size"].reshape(-1, 1)

    X = np.concatenate([dataset["X"], dataset["size"]], axis=1)

    logging.info("Sampling hyperparameters for function models...")
    hypers = sample_hypers(
        X=np.concatenate(
            (X[:, :-1], ((1 - X[:, -1])**2).reshape(-1, 1)), axis=1),
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

        x = np.concatenate(
            (X[:, :-1], ((1 - X[:, -1])**2).reshape(-1, 1)), axis=1)

        regressor.compute(x, np.sqrt(noise))

        models.append(regressor)

        X_samples = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_gen_samples, dataset["X"].shape[1] + 1)
        )

        representers.append(X_samples)
        mean, cov = regressor.predict(
            dataset["y"].reshape(-1), X_samples, return_cov=True)

        means.append(mean)
        covariances.append(cov)

        logging.debug("Computing EI...")
        exp_improvement = expected_improvement(
            mean, cov, regressor.sample(X_samples))
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
            innovations.append(np.random.normal(
                size=n_gen_samples).reshape(-1, 1))

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
        cost_regressor.compute(np.concatenate(
            [dataset["X"], dataset["size"]], axis=1), np.sqrt(noise))
        cost_models.append(cost_regressor)

    logging.info("Ready to optimize Information Gain by Cost")

    starting_point = np.concatenate([
        dataset["X"][np.argmax(dataset["y"])],
        dataset["size"][np.argmax(dataset["y"])]
    ])

    return minimize(
        fun=lambda x: - information_gain_cost(
            x, cost_models, models, dataset, p_min, representers, U, Omega),
        x0=starting_point,
        method='L-BFGS-B',
        bounds=bounds,
    )


def fabolas(obj_function, prior, bounds):
    """
        Test FABOLAS with a simple "complex" function.
    """
    iterations = 10

    if prior is None:
        logging.info("Prior not available, proceed to generate it..")
        prior = generate_prior()
    # dataset = load_prior()

    prior['y'] = prior['y'].reshape(-1)

    prior["min_time"] = 0
    if min(prior["c"]) < 0:
        prior["min_time"] = np.array(min(prior["c"]))
        prior["c"] += -prior["min_time"]

    wallclock_time = time.time()
    progress = {
        "config": np.empty((0, prior["X"].shape[1] + 1)),
        "value": np.empty((0, 1)),
        "time": np.empty((0, 1)),
        "size": np.empty((0, 1))
    }

    for i in range(iterations):
        logging.info(f"---- FABOLAS: Iteration #{i+1} ----")

        # Find the next candidate
        result = None
        while result is None:
            try:
                result = get_candidate(prior, bounds)
            except ArithmeticError:
                logging.warn("Bad surrogate, trying again..")

        logging.info(f"Evaluating function at {result.x}")
        cost = time.time()
        y = obj_function(result.x)
        cost = time.time() - cost
        iteration_time = time.time() - wallclock_time

        logging.info(f"Function value: {y} ({cost}s)")

        # Save the results
        prior["X"] = np.vstack([prior["X"], result.x[:-1]])
        prior["y"] = np.append(prior["y"], np.array([y]))
        prior["size"] = np.append(prior["size"], np.array([result.x[-1]]))
        prior["c"] = np.append(prior["c"], np.array(
            [np.log10(cost) - prior["min_time"]]))

        if min(prior["c"]) < 0:
            prior["c"] += -(np.log10(cost) - prior["min_time"])
            prior["min_time"] = np.array(np.log10(cost))

        # Also update progress
        progress["config"] = np.vstack([progress["config"], result.x])
        progress["value"] = np.append(progress["value"], np.array([y]))
        progress["time"] = np.append(
            progress["time"], np.array([iteration_time]))
        progress["size"] = np.append(progress["size"], np.array([result.x[-1]]))

    prior["y_best"] = max(prior["y"])
    imax = np.argmax(prior["y"])
    prior["X_best"] = prior["X"][imax]

    return prior, progress
