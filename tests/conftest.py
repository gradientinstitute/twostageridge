"""Fixtures for pytests."""
# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.

import numpy as np
import pytest
from sklearn.utils import check_random_state

# Test constants
RANDSTATE = 99
RANDOM = check_random_state(RANDSTATE)
N = 10000


@pytest.fixture
def make_random():
    """Random number generation with a fixed random state."""
    return RANDOM


@pytest.fixture
def make_dag1D_params():
    """Make parameters for a triangle graph with 1D treatment."""
    alpha = np.array([0.9])
    gamma = np.array([[1.9]])
    beta = np.array([1.2])
    eps = 0.3
    nu = 0.3
    return alpha, gamma, beta, eps, nu


@pytest.fixture
def make_dag2D_params():
    """Make parameters for a triangle graph with 1D treatment."""
    alpha = np.array([0.9, -0.5])
    gamma = np.array([[1.9, -2.7], [-0.6, 1.4]])
    beta = np.array([1.2, 3.2])
    eps = 0.3
    nu = 0.3
    return alpha, gamma, beta, eps, nu


def make_data(alpha, gamma, beta, eps, nu):
    """Make data for simple triangle graph."""
    D = len(alpha)
    X = RANDOM.randn(N, D)
    Z = -2. + X @ gamma + RANDOM.randn(N, D) * eps
    Y = Z @ alpha + X @ beta + RANDOM.randn(N) * nu
    W = np.hstack((Z, X))
    return W, X, Y, Z


@pytest.fixture
def make_dag1D_data(make_dag1D_params):
    """Make data for a triangle graph with 1D treatment."""
    alpha, gamma, beta, eps, nu = make_dag1D_params
    return make_data(alpha, gamma, beta, eps, nu)


@pytest.fixture
def make_dag2D_data(make_dag2D_params):
    """Make data for a triangle graph with 2D treatment."""
    alpha, gamma, beta, eps, nu = make_dag2D_params
    return make_data(alpha, gamma, beta, eps, nu)
