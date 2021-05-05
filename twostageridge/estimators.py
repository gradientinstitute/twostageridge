"""Define the two-stage ridge regression estimator."""

import numpy as np
import pandas as pd

from typing import Tuple, Union, Optional
from scipy.optimize import minimize
from scipy.linalg import solve
from sklearn.base import BaseEstimator, RegressorMixin


class TwoStageRidge(BaseEstimator, RegressorMixin):
    """Two stage ridge regression for causal response surface estimation."""

    def __init__(
        self,
        treatment_col: Union[int, str],
        regulariser1: float = 0.1,
        regulariser2: float = 0.1,
        fit_intercept: bool = True,
        tol: Optional[float] = None
    ) -> None:
        """Instantiate a two-stage ridge regression estimator."""
        if not np.isscalar(treatment_col):
            raise TypeError('treatment_col must be an int or a string only!')
        self.treatment_col = treatment_col
        self.regulariser1 = regulariser1
        self.regulariser2 = regulariser2
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, W, y):
        """Fit the two-stage ridge regression estimator."""
        W = self._transform_W(W)
        X, z = self._splitW(W)

        # Stage 1
        self.beta_c = ridge_weights(X, z, self.regulariser1)
        z_hat = X @ self.beta_c

        # Stage 2 - initialisation
        beta = ridge_weights(W, y, self.regulariser2)
        self.beta = np.delete(beta, self.treatment_ind, axis=0)
        alpha_0 = beta[self.treatment_ind]
        beta_d_0 = self.beta + alpha_0 * self.beta_c
        params_0 = np.concatenate([[alpha_0], beta_d_0])

        # Stage 2 - objective function
        r = z - z_hat

        def objective(params):
            alpha, beta_d = params[0], params[1:]
            y_hat = alpha * r + X @ beta_d
            obj = np.sum((y - y_hat)**2) \
                + self.regulariser2 * beta_d.T @ beta_d
            return obj

        # Stage 2 - gradient function
        rTr = r.T @ r
        rTy = r.T @ y
        rTX = r.T @ X
        XTX = X.T @ X
        XTy = X.T @ y

        def gradient(params):
            alpha, beta_d = params[0], params[1:]
            dalpha = 2 * (alpha * rTr - rTy + rTX @ beta_d)
            dbeta_d = 2 * (XTX @ beta_d - XTy + alpha * rTX.T
                           + self.regulariser2 * beta_d)
            dparams = np.concatenate([[dalpha], dbeta_d])
            return dparams

        # Stage 2 - solve
        res = minimize(objective, params_0, jac=gradient, method='L-BFGS-B',
                       tol=self.tol)
        self.alpha = res.x[0]
        self.beta_d = res.x[1:]
        self.coef_ = res.x
        return self

    def predict(self, W):
        """Use the two-stage ridge regression estimator for prediction."""
        W = self._transform_W(W)
        X, z = self._splitW(W)

        # Stage 1
        z_hat = X @ self.beta_c

        # Stage 2
        y_hat = self.alpha * (z - z_hat) + X @ self.beta_d
        return y_hat

    def get_params(self, deep=True):
        """Get this estimators initialisation parameters."""
        return {
            'treatment_col': self.treatment_col,
            'regulariser1': self.regulariser1,
            'regulariser2': self.regulariser2,
            'fit_intercept': self.fit_intercept,
            'tol': self.tol
        }

    def set_params(self, **parameters):
        """Set this estimators initialisation parameters."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _splitW(self, W: np.array) -> Tuple[np.array, np.array]:
        z = W[:, self.treatment_ind]
        X = np.delete(W, self.treatment_ind, axis=1)
        return X, z

    def _transform_W(self, W: np.array) -> np.array:
        if isinstance(W, pd.DataFrame):
            self.treatment_ind = list(W.columns).index(self.treatment_col)
            W = W.to_array()
        else:
            self.treatment_ind = self.treatment_col
        if self.fit_intercept:
            W = np.hstack((W, np.ones((len(W), 1))))
        return W


def ridge_weights(X: np.array, Y: np.array, gamma: float) -> np.array:
    """Compute ridge regression weights."""
    N, D = X.shape
    A = X.T @ X + np.diag(np.full(shape=D, fill_value=gamma))
    b = X.T @ Y
    weights = solve(A, b, assume_a='pos')
    return weights
