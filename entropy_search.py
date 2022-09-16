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
#from tensorflow_probability import distributions as tfp
from horseshoe import Horseshoe
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  WhiteKernel
from ard import AutomaticRelevanceDetermination
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
from tensorflow.keras.datasets import mnist
import math
import logging

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
    kernel = cov * AutomaticRelevanceDetermination(nu=5/2) + WhiteKernel()
    hyper_distribution = GaussianProcessRegressor(kernel=kernel)
    hyper_distribution.fit(X, y)

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
        #prior += tfp.Horseshoe(scale=0.1).log_prob(noise).numpy()
        prior += Horseshoe(scale=0.1).logpdf(noise)

        # Compute log likelihood of the current distribution with proposed values

        prob = hyper_distribution.log_marginal_likelihood(theta=[cov, lamb, noise])

        return prior + prob

    nwalkers, ndim, iterations = K, kernel.n_dims, 1000
    sampler = EnsembleSampler(nwalkers, ndim, log_prob)
    state = sampler.run_mcmc(np.random.rand(nwalkers, ndim), 100, rstate0=np.random.get_state())
    sampler.reset()
    sampler.run_mcmc(state, iterations)
    return sampler.chain[:, -1]

def expected_improvement(mean, covariance, y_values, exploration=0, ):
    y_max = y_values.max()

    # Since we need sigma(x) we just use the diagonal
    variance = np.sqrt(np.diag(covariance))

    u = (mean - y_max - exploration) / variance
    ei = variance * (u * sts.norm.cdf(u) + sts.norm.pdf(u))

    ei[variance <= 0.] = 0.

    return ei

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

def compute_innovations(x, model, representer_points, variance):
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
    var_noise = np.array([[10**(-4)]]) # the noise is estimated as very little in comparison to the GP variance
    # Compute the correlation matrix and get the element corresponding to x
    _, correlation_x_r = model.predict(np.concatenate((x, representer_points)), return_cov=True)
    correlation_x_r = (correlation_x_r[-1, :-1]).reshape(-1, 1) # vector (n_rep , 1)
    corr_x_r_variance = np.dot(correlation_x_r, np.linalg.inv(variance))
    d_mu = np.dot(corr_x_r_variance, np.linalg.cholesky(variance + var_noise))
    d_sigma = corr_x_r_variance * correlation_x_r.T

    return d_mu, d_sigma

def entropy_search(dataset):
    n_hyper_samples = 20    # K parameter
    n_gen_samples = 50      # Z parameter
    n_innovations = 20      # P parameter
    
    logging.info("Sampling hypeparameters..")
    # K samples with mcmc over GP hyperparameters: lambda, covariance amplitude for Matérn kernel + noise variance
    hyperparameters = sample_hypers(dataset["X"], dataset["y"], K=n_hyper_samples)
    

    Omega = []
    p_min = []
    U = []
    models = []
    representers = []
    means = []
    covariances = []

    for hyper in hyperparameters:
        kernel_cov, lamb, noise = hyper # Convert from log scale
        kernel = kernel_cov * AutomaticRelevanceDetermination(length_scale=math.e**lamb, nu=5/2) + WhiteKernel(noise_level=abs(noise))
        regressor = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(dataset["X"], dataset["y"])
        models.append(regressor)

        # sample Z point from M and get predictive mean + covariance
        X_samples = np.random.uniform(low=-10, high=10, size=(n_gen_samples, dataset["X"].shape[1])) # FIXME: sample from the parameter space, using custom function
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
        p_min.append(compute_pmin(mean, cov, derivatives=True))
        
        # generate P noise vectors from a Gaussian(0, I_Z)
        # Q: Why save'em in memory when they can be generated on the fly?
        # A: This way the noise is the same for all IG iterations
        innovations = []
        logging.debug("Generating innovations..")
        for _ in range(n_innovations):
            # Generate a gaussian noise vector
            innovations.append(np.random.normal(size=n_gen_samples))

        Omega.append(innovations)

    def information_gain(test_point):
        """
        Returns the information gain value for `test_point`

        """
        a, errors = 0, 0
        testpoint_mu = np.zeros([len(models)])
        testpoint_var = np.zeros([len(models)])
        for i,m in enumerate(models):
            testpoint_mu[i], testpoint_var[i] = m.predict(test_point.reshape(1, -1), return_cov=True)
        
        # **Predictive** variance (Hutter, 2013 - Algorithm Runtime Prediction: Methods & Evaluation)
        # Section 4.2: Scaling to large amounts of data with approximate gaussian processes
        pred_mean_testpoint = testpoint_mu.mean()
        pred_var_testpoint = (testpoint_var + testpoint_mu**2).mean() - pred_mean_testpoint**2

        for i, model in enumerate(models):
            # TODO: can it be vectorized?
            for p in range(n_innovations):
                d_mu, d_sigma = compute_innovations(test_point.reshape(1, -1), model, representers[i], pred_var_testpoint.reshape(-1, 1))

                # Compute pmin from the updated posterior
                #q_min = compute_pmin(means[i] + d_mu.reshape(-1), covariances[i] + d_sigma)
                trace = np.sum(np.sum(np.multiply(p_min[i][3], np.reshape(
                        d_mu * d_mu.T, (1, d_mu.shape[0], d_mu.shape[0]))), 2), 1)[:, np.newaxis]
                deterministic_change = np.dot(p_min[i][2], d_sigma[np.tril(np.ones((d_sigma.shape))).astype(bool), np.newaxis]) + 1/2 * trace
                
                stochastic_change = p_min[i][1] * d_mu * Omega[i][p]
                q_min = p_min[i][0] + deterministic_change + stochastic_change

                d_entropy = - np.sum(np.exp(q_min) * (q_min + U[i])) + \
                    np.sum(np.exp(p_min[i][0]) * (p_min[i][0] + U[i]))

                if d_entropy != np.nan:
                    a += 1/n_innovations * d_entropy
                else:
                    logging.warning("Cannot compute Information Gain with this model")

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

    C = np.array([-8, 0, 8])
    gamma = np.array([-8, 0, 8])

    x_values, y_values = [], [
        0.71234,  0.71234, 0.2154, 
        0.71234, 0.71234, 0.6523, 
        0.91266, 0.81256, 0.71234
        ]

    for c in C:
        for g in gamma:
            #y = obj_function((c,g), dataset)
            x_values.append((c,g))
            #y_values.append(y)

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

        logging.info(f"Evaluating function at {result.x}")
        function_time = time.time()
        # Evaluate the function
        y = obj_function(result.x, data)
        function_time = time.time() - function_time

        performance = "-" if best_y is None else str((y / best_y - 1)*100 )
        logging.info(f"Function value: {y} ({performance}%), {function_time}s")

        # Save the results
        dataset["X"] = np.vstack([dataset["X"], result.x])
        dataset["y"] = np.append(dataset["y"], np.array([y]))

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


