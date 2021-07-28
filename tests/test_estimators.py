"""Tests for the two stage ridge regression estimators."""
# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.

import numpy as np
import pytest

from scipy.linalg import solve
from sklearn.utils.estimator_checks import check_estimator

from twostageridge import TwoStageRidge, ridge_weights
from twostageridge.estimators import _check_treatment_index


def test_valid_estimator():
    """Test the estimators obey scikit learn conventions."""
    est = TwoStageRidge(treatment_index=0)
    check_estimator(est)


def test_splitting():
    """Make treatment/control splitting is working."""
    ind = 1
    labels = np.array([0, 1, 2])
    W = np.ones((10, 3)) * labels
    Y = np.random.randn(10)

    est = TwoStageRidge(treatment_index=ind, fit_intercept=False).fit(W, Y)
    W, X, z = est._splitW(W)

    assert all(z == 1)
    assert X.shape == (10, 2)
    assert all(X[:, 0] == 0)
    assert all(X[:, 1] == 2)
    assert W.shape == (10, 3)
    assert np.all(W == W)


def test_intercept_indexing():
    """Make sure negative indexing is working with an intercept."""
    ind = -2
    labels = np.array([-3, -2, -1])
    W = np.ones((10, 3)) * labels
    Y = np.ones(10)

    est = TwoStageRidge(treatment_index=ind, fit_intercept=True).fit(W, Y)
    _, _, _ = est._splitW(W)  # Test repeated calling
    W, X, z = est._splitW(W)

    assert all(z == ind)
    assert all(W[:, -1] == 1)
    assert np.all(z == W[:, est.adjust_tind_])


@pytest.mark.parametrize('index', [-1, 1, slice(0, -2), np.array([0, 1, -1])])
@pytest.mark.parametrize('intercept', [False, True])
def test_intercept_checks(index, intercept):
    """Test the checking code for indexing."""
    labels = np.arange(5)
    W = np.ones((10, 5)) * labels

    adj_ind = _check_treatment_index(index, W, intercept)
    assert np.all(W[:, index] == W[:, adj_ind])


@pytest.mark.parametrize('params, data', [
    ('make_dag1D_params', 'make_dag1D_data'),
    ('make_dag2D_params', 'make_dag2D_data'),
])
@pytest.mark.parametrize('reg_vec', [True, False])
def test_ridge_weights(params, data, reg_vec, request):
    """Make sure ridge_weights can return an accurate estimate."""
    alpha, gamma, beta, eps, nu = request.getfixturevalue(params)
    W, X, Y, Z = request.getfixturevalue(data)
    Xint = np.hstack((X, np.ones((len(X), 1))))

    reg = 0.1 * np.ones(Xint.shape[1]) if reg_vec else 0.1
    gamma_rr, _, _ = ridge_weights(Xint, Z, gamma=reg)
    gamma_rr = gamma_rr[:-1, :]
    assert np.allclose(gamma_rr, gamma, rtol=0.01)


def test_ridge_dof(make_random):
    """Test estimated degrees of freedom for the ridge regressor."""
    N, D = 500, 250
    gamma = 100
    rand = make_random
    X = rand.randn(N, D)
    Y = X @ rand.randn(D) + rand.randn(N)

    # test OLS dof
    ols_dof = N - D
    _, dof_t, dof_s = ridge_weights(X, Y, gamma=gamma, ols_dof=True)
    assert dof_t == dof_s
    assert dof_t == ols_dof

    # test ridge dof, these should be bigger than OLS since regularisation
    # reduces the effective dimensionality of the features.
    _, dof_t, dof_s = ridge_weights(X, Y, gamma=gamma, ols_dof=False)
    assert dof_t > ols_dof
    assert dof_s > ols_dof
    assert dof_t > dof_s

    # make sure the low dimensional computations match the full dimensionality
    H = X @ solve(X.T @ X + gamma * np.eye(D), X.T, assume_a='pos')

    dof_t_full = N - np.trace(H)
    assert np.allclose(dof_t, dof_t_full)

    dof_s_full = N - np.trace(2*H - H @ H.T)
    assert np.allclose(dof_s, dof_s_full)

    # make sure we throw a warning when ols_dof cannot be computed
    Y = X.T @ rand.randn(N) + rand.randn(D)
    with pytest.warns(RuntimeWarning, match='.*D >= N.*'):
        ridge_weights(X.T, Y, gamma=gamma, ols_dof=True)


@pytest.mark.parametrize('params, data', [
    ('make_dag1D_params', 'make_dag1D_data'),
    ('make_dag2D_params', 'make_dag2D_data'),
])
def test_estimator_weights(params, data, request):
    """Make sure ridge_weights can return an accurate estimate."""
    alpha, gamma, beta, eps, nu = request.getfixturevalue(params)
    W, X, Y, Z = request.getfixturevalue(data)
    ti = slice(0, len(alpha))

    est = TwoStageRidge(treatment_index=ti, regulariser1=.1, regulariser2=.1)
    est.fit(W, Y)

    assert np.allclose(est.beta_c_[:-1, :], gamma, rtol=0.05)
    assert np.allclose(est.alpha_, alpha, rtol=0.05)

    se_alpha = np.sqrt(nu**2 / np.sum(Z**2))
    assert np.allclose(se_alpha, est.se_alpha_, atol=0.05)


@pytest.mark.parametrize('params, data', [
    ('make_dag1D_params', 'make_dag1D_data'),
    ('make_dag2D_params', 'make_dag2D_data'),
])
def test_score_treatments(params, data, request):
    """Make sure the treatment model score method is working."""
    alpha, gamma, beta, eps, nu = request.getfixturevalue(params)
    W, X, Y, Z = request.getfixturevalue(data)
    ti = slice(0, len(alpha))

    est = TwoStageRidge(treatment_index=ti, regulariser1=.1, regulariser2=.1)
    est.fit(W, Y)

    assert est.score_stage1(W) > .95


@pytest.mark.parametrize('params, data', [
    ('make_dag1D_params', 'make_dag1D_data'),
    ('make_dag2D_params', 'make_dag2D_data'),
])
def test_predict_stage1(params, data, request):
    """Make sure the stage 1 predictions are returning expected results."""
    alpha, gamma, beta, eps, nu = request.getfixturevalue(params)
    W, X, Y, Z = request.getfixturevalue(data)
    ti = slice(0, len(alpha))

    est = TwoStageRidge(treatment_index=ti, regulariser1=.1, regulariser2=.1)
    est.fit(W, Y)
    z_hat, z = est.predict_stage1(W)

    assert z_hat.shape == z.shape == Z.shape
    assert np.all(Z == z)


@pytest.mark.parametrize('params, data', [
    ('make_dag1D_params', 'make_dag1D_data'),
    ('make_dag2D_params', 'make_dag2D_data'),
])
def test_stats(params, data, request):
    """Make sure the computed statistics are the correct dimensions."""
    alpha, gamma, beta, eps, nu = request.getfixturevalue(params)
    W, X, Y, Z = request.getfixturevalue(data)
    ti = slice(0, len(alpha))

    est = TwoStageRidge(treatment_index=ti, regulariser1=.1, regulariser2=.1)
    est.fit(W, Y)
    stats = est.model_statistics()

    alpha = np.squeeze(alpha)
    assert np.shape(stats.alpha) == np.shape(alpha)
    assert np.shape(stats.std_err) == np.shape(alpha)
    assert np.shape(stats.p_value) == np.shape(alpha)
    assert np.shape(stats.t_stat) == np.shape(alpha)
    assert np.isscalar(stats.dof)
