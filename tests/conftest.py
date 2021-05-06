"""Fixtures for pytests."""
import pytest
import numpy as np

from sklearn.utils import check_random_state


# Test constants
RANDSTATE = 99
RANDOM = check_random_state(RANDSTATE)
N = 300


@pytest.fixture
def make_random():
    return RANDOM


@pytest.fixture
def make_triangle_dag_params():
    alpha = 0.9
    gamma = 1.9
    beta = 1.2
    eps = 0.1
    nu = 0.3
    return alpha, gamma, beta, eps, nu


@pytest.fixture
def make_triangle_dag_data(make_triangle_dag_params):
    alpha, gamma, beta, eps, nu = make_triangle_dag_params
    X = RANDOM.randn(N)
    Z = -2. + gamma * X + RANDOM.randn(N) * eps
    Y = alpha * Z + beta * X + nu * RANDOM.randn(N) * nu
    X = X[:, np.newaxis]
    W = np.hstack((Z[:, np.newaxis], X))
    return W, X, Y, Z
