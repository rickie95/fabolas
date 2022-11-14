import logging
import time

import numpy as np
import scipy.stats as sts
from emcee import EnsembleSampler
from scipy.optimize import minimize
from george.gp import GP
from george.kernels import Matern52Kernel

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
    kernel = cov * Matern52Kernel(
            metric=np.ones(X.shape[1]),
            ndim=X.shape[1],
            axes=[x for x in range(X.shape[1])]
        )
    hyper_distribution = GP(kernel=kernel, mean=np.mean(y))
    hyper_distribution.compute(X, yerr=0.05)

    def log_prob(theta):
        """
        Loglikelihood of the current distribution over hypers + prior
        """
        cov = theta[0]
        lamb = theta[1:-1]
        noise = theta[-1]

        prior = 0

        # Check first if all parameters are inside some reasonable bounds.
        # This is mostly due to avoid silly choices that could be ammissible
        # but numerically useless.

        if not 0 < cov < 100:
            return -np.inf

        # Log likelihood for lambda is 0 when lambda is inside its bounds
        if not (np.all(np.array(lamb) > -9) and np.all(np.array(lamb) < 2)):
            return -np.inf

        if not -20 < noise < 20:
            return -np.inf

        # Log probability for lognormal distribution
        prior += sts.lognorm.logpdf(cov, covariance_prior_mean,
                                    covariance_prior_sigma)

        # Horseshoe
        prior += Horseshoe(scale=0.1).logpdf(noise)

        hyper_distribution.kernel.set_parameter_vector([cov, *lamb])
        try:
            hyper_distribution.compute(X, yerr=np.sqrt(noise))
        except:
            return -np.inf

        return prior + hyper_distribution.log_likelihood(y if len(y.shape) == 1 else y.reshape(-1), quiet=True)

    nwalkers, ndim, iterations = K, kernel.ndim + 2, 500
    sampler = EnsembleSampler(nwalkers, ndim, log_prob)
    state = sampler.run_mcmc(np.random.rand(
        nwalkers, ndim), 100, rstate0=np.random.get_state())
    sampler.reset()
    sampler.run_mcmc(state, iterations)
    return sampler.chain[:, -1]


def entropy_search(dataset, bounds):
    n_hyper_samples = 20    # K parameter
    n_gen_samples = 50      # Z parameter
    n_innovations = 20      # P parameter

    logging.info("Sampling hypeparameters..")
    # K samples with mcmc over GP hyperparameters:
    # - lambda
    # - covariance amplitude for Matérn kernel
    # - noise variance
    hyperparameters = sample_hypers(
        dataset["X"], dataset["y"], K=n_hyper_samples)

    Omega = []
    p_min = []
    U = []
    models = []
    representers = []
    means = []
    covariances = []

    for hyper in hyperparameters:
        kernel_cov = hyper[0]
        lamb = hyper[1:-1]
        noise = hyper[-1]
        kernel = kernel_cov * Matern52Kernel(
            metric=np.e**lamb,
            ndim=dataset["X"].shape[1],
            axes=[x for x in range(dataset["X"].shape[1])]
        )
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
        mean, cov = regressor.predict(dataset["y"].reshape(-1), X_samples, return_cov=True)
        means.append(mean)
        covariances.append(cov)
        # compute their Expected Improvement
        logging.debug("Computing EI...")
        exp_improvement = expected_improvement(
            mean, cov, regressor.sample(X_samples))
        U.append(exp_improvement)

        # Compute p_min using EPMGP
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

    # maximize information gain => minimize -information_gain()
    # FIXME: find a better strategy for the initial guess/guesses.
    # Maybe random + last good configuration?
    logging.info("Ready to optimize Information Gain")
    return minimize(
        fun=lambda x: -
        information_gain(x, models, p_min, representers, U, Omega, dataset),
        x0=dataset["X"][np.argmax(dataset["y"])],
        method='L-BFGS-B',
        bounds=bounds,
    )


def es(obj_function, prior, bounds):
    """
        Perform bayesian optimization of `obj_function` over `data`, using 
        `prior` as baseline.
    """
    iterations = 10

    # The stopping criteria is based on a fixed number
    # of iterations but could take in account
    # a "min improvement" policy

    wallclock_time = time.time()
    progress = {
        "config": np.empty((0, prior["X"].shape[1])),
        "value": np.empty((0, 1)),
        "time": np.empty((0, 1))
    }

    for i in range(iterations):
        logging.info(f"---- ES: Iteration #{i+1} ----")

        # Find the next candidate
        result = None
        while result is None:
            try:
                result = entropy_search(prior, bounds)
            except ArithmeticError:
                logging.warn("Bad surrogate, trying again..")

        logging.info(f"Evaluating function at {result.x}")
        # Evaluate the function
        y = obj_function(result.x)
        iteration_time = time.time() - wallclock_time

        logging.info(f"Function value: {y} ({iteration_time}s)")

        # Save the results
        prior["X"] = np.vstack([prior["X"], result.x])
        prior["y"] = np.vstack([prior["y"], np.array([y])])

        # Also update progress
        progress["config"] = np.vstack([progress["config"], result.x])
        progress["value"] = np.vstack([progress["value"], np.array([y])])
        progress["time"] = np.vstack([
            progress["time"], np.array([iteration_time])])

    prior["y_best"] = max(prior["y"])
    imax = np.argmax(prior["y"])
    prior["X_best"] = prior["X"][imax]

    return prior, progress
