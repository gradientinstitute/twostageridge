"""Tests for the two stage ridge regression estimators."""
import numpy as np
import pytest

from sklearn.utils.estimator_checks import check_estimator

from twostageridge import TwoStageRidge, ridge_weights
from conftest import (make_dag1D_data, make_dag2D_data, make_dag1D_params,
                      make_dag2D_params)


def test_valid_estimator():
    """Test the estimators obey scikit learn conventions."""
    est = TwoStageRidge(treatment_index=0)
    check_estimator(est)


def test_splitting():
    """Make treatment/control splitting is working."""
    ind = 1
    labels = np.array([0, 1, 2])
    W = np.ones((10, 3)) * labels
    Y = np.ones(10)

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
    assert all(z == W[:, est._alphaind])


@pytest.mark.parametrize('params, data', [
    (make_dag1D_params(), make_dag1D_data()),
    (make_dag2D_params(), make_dag2D_data()),
])
def test_ridge_weights(params, data):
    """Make sure ridge_weights can return an accurate estimate."""
    alpha, gamma, beta, eps, nu = params
    W, X, Y, Z = data

    Xint = np.hstack((X, np.ones((len(X), 1))))
    gamma_rr = ridge_weights(Xint, Z, gamma=.1)[:-1]
    assert np.allclose(gamma_rr, gamma, rtol=0.01)


@pytest.mark.parametrize('params, data', [
    (make_dag1D_params(), make_dag1D_data()),
    # (make_dag2D_params(), make_dag2D_data()),
])
def test_estimator_weights(params, data):
    """Make sure ridge_weights can return an accurate estimate."""
    alpha, gamma, beta, eps, nu = params
    W, X, Y, Z = data

    # treatment_index = 
    est = TwoStageRidge(treatment_index=0, regulariser1=.1, regulariser2=.1)
    est.fit(W, Y)

    assert np.allclose(est.beta_c_[0], gamma, rtol=0.02)
    assert np.allclose(est.alpha_, alpha, rtol=0.02)

    se_alpha = np.sqrt(nu**2 / np.sum(Z**2))
    assert np.allclose(se_alpha, est.se_alpha_, atol=0.02)
