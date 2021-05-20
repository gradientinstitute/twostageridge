"""Define the two-stage ridge regression estimator."""

import numpy as np
from typing import Tuple, Optional, TypeVar, Union
from functools import singledispatch

from scipy.linalg import solve
from scipy.stats import t

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score


# Make a return type for "self"
Self = TypeVar('Self', bound='TwoStageRidge')


#
# Public classes and functions
#

class TwoStageRidge(BaseEstimator, RegressorMixin):
    """Two stage ridge regression for causal response surface estimation."""

    def __init__(
        self,
        treatment_index: Union[int, np.array, slice] = 0,
        regulariser1: float = 0.1,
        regulariser2: float = 0.1,
        fit_intercept: bool = True
    ) -> None:
        """Instantiate a two-stage ridge regression estimator."""
        if (regulariser1 < 0) or (regulariser2 < 0):
            raise ValueError('regulariser coefficients must have a value >= 0')

        self.treatment_index = treatment_index
        self.regulariser1 = regulariser1
        self.regulariser2 = regulariser2
        self.fit_intercept = fit_intercept

    def fit(self, W: np.ndarray, y: np.ndarray) -> Self:
        """Fit the two-stage ridge regression estimator."""
        # Checks and input transforms
        W, y = check_X_y(W, y, y_numeric=True)
        self.adjust_tind_ = _check_treatment_index(self.treatment_index, W,
                                                   self.fit_intercept)
        self.n_features_in_ = W.shape[1]  # required to be sklearn compatible
        W, X, z = self._splitW(W)

        # Stage 1
        self.beta_c_ = ridge_weights(X, z, self.regulariser1)
        z_hat = X @ self.beta_c_

        # Stage 2 - Make the z-columns residual columns
        r = z - z_hat
        Wres = W.copy()
        Wres[:, self.adjust_tind_] = np.squeeze(r) if \
            np.isscalar(self.adjust_tind_) else r

        # Stage 2 - Only regularise non-treatment weights
        N, D = W.shape
        reg2_diag = np.ones(D) * self.regulariser2
        reg2_diag[self.adjust_tind_] = 0.

        # Stage 2 - Compute the weights
        weights = ridge_weights(Wres, y, gamma=reg2_diag)
        self.alpha_ = np.atleast_1d(weights[self.adjust_tind_])
        self.beta_d_ = np.delete(weights, self.adjust_tind_, axis=0)

        # Compute alpha standard error (OLS), t-statistic and p-value
        eps = y - (r @ self.alpha_ + X @ self.beta_d_)
        s2 = np.sum(eps**2) / (N - D)
        rTr_diag = np.sum(r**2, axis=0)
        self.se_alpha_ = np.sqrt(s2 / rTr_diag)
        self.t_ = self.alpha_ / self.se_alpha_  # h0 is alpha = 0
        self.p_ = (1. - t.cdf(np.abs(self.t_), df=N - D)) * 2  # h1 != 0
        return self

    def predict(self, W: np.ndarray) -> np.ndarray:
        """Use the two-stage ridge regression estimator for prediction."""
        check_is_fitted(self, attributes=['alpha_', 'beta_d_'])
        W = check_array(W)
        _, X, z = self._splitW(W)

        # Stage 1
        z_hat = X @ self.beta_c_

        # Stage 2
        y_hat = (z - z_hat) @ self.alpha_ + X @ self.beta_d_
        return y_hat

    def score_stage1(
        self,
        W: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Get the R^2 score of the stage 1 regression model."""
        check_is_fitted(self, attributes=['alpha_', 'beta_d_'])
        W = check_array(W)
        _, X, z = self._splitW(W)
        z_hat = X @ self.beta_c_
        r2 = r2_score(z, z_hat, sample_weight=sample_weight)
        return r2

    def get_params(self, deep: bool = True) -> dict:
        """Get this estimators initialisation parameters."""
        return {
            'treatment_index': self.treatment_index,
            'regulariser1': self.regulariser1,
            'regulariser2': self.regulariser2,
            'fit_intercept': self.fit_intercept,
        }

    def set_params(self, **parameters: dict):
        """Set this estimators initialisation parameters."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _splitW(self, W: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split W into X and z, add an intercept term to W, X optionally."""
        z = W[:, self.treatment_index]
        if np.ndim(z) == 1:
            z = z[:, np.newaxis]
        if self.fit_intercept:
            W = np.hstack((W, np.ones((len(W), 1))))
        X = np.delete(W, self.adjust_tind_, axis=1)
        return W, X, z


def ridge_weights(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Union[float, np.ndarray]
) -> np.ndarray:
    """Compute ridge regression weights."""
    N, D = X.shape
    if np.isscalar(gamma):
        gamma_diag = np.full(shape=D, fill_value=gamma)
    else:
        if gamma.shape != (D,):
            raise TypeError('gamma has to be a scalar or vector of X.shape[1]')
        gamma_diag = gamma

    A = X.T @ X + np.diag(gamma_diag)
    b = X.T @ Y
    weights = solve(A, b, assume_a='pos')
    return weights


#
# Private module utilities
#

@singledispatch
def _check_treatment_index(
    treatment_index,
    W: np.ndarray,
    fit_intercept: bool
) -> Union[int, np.array, slice]:
    """Check for a valid treatment index into W."""
    raise TypeError('treatment_index must be an int, and array of int,'
                    ' or a slice.')


@_check_treatment_index.register
def _(treatment_index: int, W: np.ndarray, fit_intercept: bool) -> int:
    D = W.shape[1]

    if (treatment_index >= D) or (treatment_index < -D):
        raise ValueError('treatment_index is out of bounds.')

    # Make sure adjust_tind_ indexes right weights for initialisation
    adjust_tind = treatment_index
    if fit_intercept and (treatment_index < 0):
        adjust_tind = D + treatment_index

    return adjust_tind


@_check_treatment_index.register
def _(treatment_index: slice, W: np.ndarray, fit_intercept: bool) -> slice:
    D = W.shape[1]

    if (treatment_index.start >= D) \
            or (treatment_index.start < -D) \
            or (treatment_index.stop >= D) \
            or (treatment_index.stop < -D):
        raise ValueError('treatment_index slice is out of bounds.')

    start, stop = treatment_index.start, treatment_index.stop

    # Make sure adjust_tind_ indexes right weights for initialisation
    if fit_intercept and (treatment_index.start < 0):
        start = D + treatment_index.start
    if fit_intercept and (treatment_index.stop < 0):
        stop = D + treatment_index.stop

    adjust_tind = slice(start, stop, treatment_index.step)
    return adjust_tind


@_check_treatment_index.register
def _(treatment_index: np.ndarray, W: np.ndarray, fit_intercept: bool) \
        -> np.ndarray:
    D = W.shape[1]

    adjust_tind = np.copy(treatment_index)
    for n, i in enumerate(treatment_index):
        if (i >= D) or (i < -D):
            raise ValueError(f'treatment_index {i} is out of bounds.')

        # Make sure adjust_tind_ indexes right weights for initialisation
        if fit_intercept and (i < 0):
            adjust_tind[n] = D + treatment_index[n]

    return adjust_tind
