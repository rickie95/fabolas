import numpy as np
import scipy.stats as sts

import logging


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
    # the noise is estimated as very little in comparison to the GP variance
    var_noise = np.array([[10**(-4)]])
    # Compute the correlation matrix and get the element corresponding to x
    _, correlation_x_r = model.predict(np.concatenate((x, representer_points)), return_cov=True)
    correlation_x_r = (correlation_x_r[-1, :-1]).reshape(-1, 1)   # vector (n_rep , 1)
    corr_x_r_variance = np.dot(correlation_x_r, np.linalg.inv(variance))
    d_mu = np.dot(corr_x_r_variance, np.linalg.cholesky(variance + var_noise))
    d_sigma = corr_x_r_variance * correlation_x_r.T

    return d_mu, d_sigma


def expected_improvement(mean: np.array, covariance: np.array, y_values: np.array, exploration=0):
    y_max = y_values.max()

    # Since we need sigma(x) we just use the diagonal
    variance = np.sqrt(np.diag(covariance))

    u = (mean - y_max - exploration) / variance
    ei = variance * (u * sts.norm.cdf(u) + sts.norm.pdf(u))

    ei[variance <= 0.] = 0.

    return ei


def information_gain(test_point, models, p_min, representers, U, Omega, n_innovations=20):
    """
    Returns the information gain value for `test_point`

    """
    a = 0
    testpoint_mu = np.zeros([len(models)])
    testpoint_var = np.zeros([len(models)])
    for i, m in enumerate(models):
        testpoint_mu[i], testpoint_var[i] = m.predict(test_point.reshape(1, -1), return_cov=True)

    while np.isnan(testpoint_mu.sum()) or np.isnan(testpoint_var.sum()):
        faulty = [x or y for x, y in zip(np.isnan(testpoint_mu), np.isnan(testpoint_var))]
        for i, m in enumerate(models):
            if faulty[i]:
                testpoint_mu[i], testpoint_var[i] = m.predict(test_point.reshape(1, -1),
                                                              return_cov=True)

    # **Predictive** variance (Hutter, 2013 - Algorithm Runtime Prediction: Methods & Evaluation)
    # Section 4.2: Scaling to large amounts of data with approximate gaussian processes
    pred_mean_testpoint = testpoint_mu.mean()
    pred_var_testpoint = (testpoint_var + testpoint_mu**2).mean() - pred_mean_testpoint**2

    for i, model in enumerate(models):
        # TODO: can it be vectorized?
        for p in range(n_innovations):
            d_mu, d_sigma = compute_innovations(test_point.reshape(1, -1), model, representers[i],
                                                pred_var_testpoint.reshape(-1, 1))

            # Compute pmin from the updated posterior
            # q_min = compute_pmin(means[i] + d_mu.reshape(-1), covariances[i] + d_sigma)
            trace = np.sum(np.sum(np.multiply(p_min[i][3], np.reshape(
                    d_mu * d_mu.T, (1, d_mu.shape[0], d_mu.shape[0]))), 2), 1)[:, np.newaxis]
            upper_tri_indxs = np.tril(np.ones((d_sigma.shape))).astype(bool)
            deterministic_change = np.dot(p_min[i][2], d_sigma[upper_tri_indxs, np.newaxis]) + \
                1/2 * trace

            stochastic_change = p_min[i][1] * d_mu * Omega[i][p]
            q_min = p_min[i][0] + deterministic_change + stochastic_change

            d_entropy = - np.sum(np.exp(q_min) * (q_min + U[i])) + \
                np.sum(np.exp(p_min[i][0]) * (p_min[i][0] + U[i]))

            if d_entropy != np.nan:
                a += 1/n_innovations * d_entropy
            else:
                logging.warning("Cannot compute Information Gain with this model")

    logging.info(f"IG: {1/len(models) * a} for test point: {test_point}")
    return - (1/len(models) * a)


def information_gain_cost(test_point, cost_model, models, p_min, representers, U, Omega):
    overhead_cost = 0.5
    predicted_cost = cost_model.sample(test_point)
    cost_factor = 1/(predicted_cost + overhead_cost)
    return cost_factor * information_gain(test_point, models, p_min, representers, U, Omega)
