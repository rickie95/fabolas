import numpy as np
import scipy.stats as sts

import logging


def compute_innovations(x, y, model, representer_points, variance):
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
        - `d_mu`: np.array(P,1)
        - `d_sigma`: np.array(P, P)
    """
    # the noise is estimated as very little in comparison to the GP variance
    var_noise = np.array([[10**(-4)]])
    # Compute the correlation matrix and get the element corresponding to x
    # shape must be (P + 1, P + 1)
    # _, correlation_x_r = model.predict(np.concatenate((x, representer_points)), return_cov=True)
    _, correlation_x_r = model.predict(y.reshape(-1), np.concatenate((x, representer_points)), return_cov=True)
    correlation_x_r = (correlation_x_r[-1, :-1]).reshape(-1, 1)   # vector (n_rep , 1)
    corr_x_r_variance = np.dot(correlation_x_r, np.linalg.inv(variance))
    d_mu = np.dot(corr_x_r_variance, np.linalg.cholesky(variance + var_noise))
    d_sigma = corr_x_r_variance * correlation_x_r.T

    return d_mu, d_sigma


def expected_improvement(mean: np.array, covariance: np.array, y_values: np.array, exploration=0):
    """
        Compute Expected Improvement

        ### Parameters
        - mean:         np.array (N,)
        - covariance:   np.array (N, N)
        - y_values:     np.array (N,)

        ### Returns
        - ei:           np.array(N,)
    """
    y_max = y_values.max()

    # Double check mean array, in some versions of numpy
    # a 2D array would produce an NxN result which
    #  would be incorrect.
    if len(mean.shape) > 1:
        mean = mean.reshape(-1)

    # Since we need sigma(x) we just use the diagonal
    variance = np.sqrt(np.diag(covariance))

    assert variance.shape == (mean.shape[0],)

    logging.debug(f"Variance shape {variance.shape}")

    u = (mean - y_max - exploration) / variance
    ei = variance * (u * sts.norm.cdf(u) + sts.norm.pdf(u))

    assert u.shape == (mean.shape[0],)
    assert ei.shape == (mean.shape[0],)

    ei[variance <= 0.] = 0.

    return ei


def predict_testpoint(models, test_point):
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

    return pred_mean_testpoint, pred_var_testpoint


def predict_testpoint_george(models, y, test_point):
    testpoint_mu = np.zeros([len(models)])
    testpoint_var = np.zeros([len(models)])
    for i, m in enumerate(models):
        testpoint_mu[i], testpoint_var[i] = m.predict(y.reshape(-1), test_point.reshape(1, -1))

    while np.isnan(testpoint_mu.sum()) or np.isnan(testpoint_var.sum()):
        faulty = [x or y for x, y in zip(np.isnan(testpoint_mu), np.isnan(testpoint_var))]
        for i, m in enumerate(models):
            if faulty[i]:
                testpoint_mu[i], testpoint_var[i] = m.predict(y, test_point.reshape(1, -1))

    # **Predictive** variance (Hutter, 2013 - Algorithm Runtime Prediction: Methods & Evaluation)
    # Section 4.2: Scaling to large amounts of data with approximate gaussian processes
    pred_mean_testpoint = testpoint_mu.mean()
    pred_var_testpoint = (testpoint_var + testpoint_mu**2).mean() - pred_mean_testpoint**2

    return pred_mean_testpoint, pred_var_testpoint


def information_gain(test_point, models, p_min, representers, U, Omega, dataset, n_innovations=20,
                     enable_log=True):
    """
    Returns the information gain value for `test_point`

    """
    a = 0
    _, pred_var_testpoint = predict_testpoint_george(models, dataset["y"], test_point)
    for i, model in enumerate(models):
        # TODO: can it be vectorized?
        for p in range(n_innovations):
            d_mu, d_sigma = compute_innovations(test_point.reshape(1, -1), dataset["y"], model,
                                                representers[i], pred_var_testpoint.reshape(-1, 1))

            # Compute pmin from the updated posterior
            # q_min = compute_pmin(means[i] + d_mu.reshape(-1), covariances[i] + d_sigma)
            trace = np.sum(np.sum(np.multiply(p_min[i][3], np.reshape(
                    d_mu * d_mu.T, (1, d_mu.shape[0], d_mu.shape[0]))), 2), 1)[:, np.newaxis]
            upper_tri_indxs = np.tril(np.ones((d_sigma.shape))).astype(bool)
            deterministic_change = np.dot(p_min[i][2], d_sigma[upper_tri_indxs, np.newaxis]) + \
                1/2 * trace

            stochastic_change = np.dot(np.dot(p_min[i][1], d_mu), Omega[i][p].T)  # Shape: (P, P)
            q_min = p_min[i][0] + deterministic_change + stochastic_change

            # The approximate prob q_min comes as a (P, P) matrix and
            # needs to be dealt accordingly.
            # First must be normalized:

            q_max_value = np.amax(q_min, axis=0)  # np.array((P,))
            q_log = q_max_value + np.log(np.sum(np.exp(q_min - q_max_value), axis=0))
            norm = q_max_value if np.any(np.isinf(q_log)) else q_log
            q_min = np.subtract(q_min, norm)

            d_entropy = np.sum(np.exp(p_min[i][0]) * (p_min[i][0] + np.log(U[i])))

            d_entropy -= np.sum(np.multiply(
                                np.exp(q_min),
                                np.add(q_min, np.log(U[i]))),
                                axis=0)

            if not np.any(np.isnan(d_entropy)):
                a += 1/n_innovations * d_entropy.mean()
            else:
                logging.warning("Cannot compute Information Gain with this model")
                raise ArithmeticError("Cannot compute Information Gain with this model")
    if enable_log:
        logging.info(f"IG: {1/len(models) * a} for test point: {test_point}")
    return 1/len(models) * a


def information_gain_cost(test_point, cost_models, models, dataset, p_min, representers, U, Omega, n_innovations=20):
    overhead_cost = 0.0001
    predicted_cost, _ = predict_testpoint_george(
        cost_models, 
        dataset["c"], 
        test_point
        )
    cost_factor = 1/(predicted_cost + overhead_cost)
    ig = information_gain(test_point, models, p_min,
                          representers, U, Omega, dataset, n_innovations, enable_log=False)
    ig_cost = cost_factor * ig
    logging.info(
        f"IG: {ig}, cost_f {cost_factor}, ig_cost {ig_cost}. x = {test_point}")
    return ig_cost