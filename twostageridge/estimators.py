"""Define the two-stage ridge regression estimator."""
# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.

import numpy as np
from typing import Tuple, Optional, TypeVar, Union, NamedTuple
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

class StatisticalResults(NamedTuple):
    """Statistical results object.

    Attributes
    ----------
    alpha: float or ndarray
        The estimated effect size(s) (ATE) for each treatment.
    std_err: float or ndarray
        The standard error of the estimated effect size(s) (ATE) for each
        treatment.
    t_stat: float or ndarray
        The t-statistics for the estimated effect size(s) (ATE) for each
        treatment.
    p_value: float or ndarray
        The p-value of the two-sided t-test on the treatment effects.  The null
        hypothesis is that alpha = 0, and the alternate hypothesis is alpha !=
        0.
    dof: int
        The degrees of freedom used to compute the standard error and the
        t-test.
    """

    alpha: Union[float, np.ndarray]
    std_err: Union[float, np.ndarray]
    t_stat: Union[float, np.ndarray]
    p_value: Union[float, np.ndarray]
    dof: int

    def __repr__(self) -> str:
        """Return string representation of StatisticalResults."""
        reprs = f"""Statistical results:
            alpha =
                {self.alpha},
            s.e.(alpha) =
                {self.std_err}
            t-statistic(s):
                {self.t_stat}
            p-value(s):
                {self.p_value}
            Degrees of freedom: {self.dof}
            """
        return reprs


class TwoStageRidge(BaseEstimator, RegressorMixin):
    """Two stage ridge regression for causal response surface estimation.

    Parameters
    ----------
    treatment_index : int, ndarray or slice
        The column-index/indices into the covariates, W, indicating where the
        treatment variable(s) are located.
    regulariser1 : float
        The regulariser coefficient over the first stage model weights. The
        first stage is the treatment selection model. The first stage model
        predicts the treatment variables from the control variables.
    regulariser2 : float
        The regularisation coefficient over the second stage model weights for
        the control variables. The second stage model predicts the outcome
        variable from the treatment variable prediction error (from the first
        stage) and the control variables.
    fit_intercept : bool
        Fit and intercept term on the first and second stage models. This will
        append a column of ones onto the model covariates, W.

    Attributes
    ----------
    alpha_ : float or ndarray
        The treatment effect coefficient(s).
    beta_c_ : ndarray
        The first stage model weights predicting the treatment.
    beta_d_ : ndarray
        The second stage model weights predicting the treatment from the
        controls, X.
    se_alpha_ : float or ndarray
        The standard error(s) of alpha_.
    t_ : float or ndarray
        The t-statistic(s) of alpha_ with N - D degrees of freedom.
    p_ : float or ndarray
        The p-value(s) of alpha_ from a two-sided t-test with the null
        hypothesis being alpha_ = 0.

    Note
    ----
    The p-values computed are probably conservative. The actual degrees of
    freedom is probably greater than N - D, since the effective number of
    dimensions used is less than D. See the following reference for more
    information.

    Cule, E., Vineis, P. & De Iorio, M. Significance testing in ridge
    regression for genetic data. BMC Bioinformatics 12, 372 (2011).
    https://doi.org/10.1186/1471-2105-12-372
    """

    def __init__(
        self,
        *,
        treatment_index: Union[int, np.ndarray, slice] = 0,
        regulariser1: float = 1.,
        regulariser2: float = 1.,
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
        """Fit the two-stage ridge regression estimator.

        This will compute the treatment effect and store it in the `alpha_`
        object attribute.

        Parameters
        ----------
        W : ndarray
            The `(N, D)` model covariates - which includes the controls *and*
            the treatment variables. The treatment variables should be indexed
            by `treatment_index` passed into this classes' constructor.
        y : ndarray
            The `(N,)` array of outcomes.
        """
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
        #  I think this is more-or-less valid. We may have fewer degrees of
        #  freedom though (see doi: 10.1186/1471-2105-12-372).
        eps = y - (r @ self.alpha_ + X @ self.beta_d_)
        self.dof_ = max(N - D, 1)  # degrees of freedom, this is conservative
        s2 = np.sum(eps**2) / self.dof_
        rTr_diag = np.sum(r**2, axis=0)
        self.se_alpha_ = np.sqrt(s2 / rTr_diag)
        self.t_ = self.alpha_ / self.se_alpha_  # h0 is alpha = 0
        self.p_ = (1. - t.cdf(np.abs(self.t_), df=self.dof_)) * 2  # h1 != 0
        return self

    def predict(self, W: np.ndarray) -> np.ndarray:
        """Use the two-stage ridge regression estimator for prediction.

        This method is mainly useful for model selection.

        Parameters
        ----------
        W : ndarray
            The `(N, D)` model covariates - which includes the controls *and*
            the treatment variables. The treatment variables should be indexed
            by `treatment_index` passed into this classes' constructor.

        Returns
        -------
        y_hat : ndarray
            The `(N,)` array of predicted outcomes (from the second stage
            model).
        """
        check_is_fitted(self, attributes=['alpha_', 'beta_d_'])
        W = check_array(W)
        _, X, z = self._splitW(W)

        # Stage 1
        z_hat = X @ self.beta_c_

        # Stage 2
        y_hat: np.ndarray = (z - z_hat) @ self.alpha_ + X @ self.beta_d_
        return y_hat

    def predict_stage1(self, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the treatments using only the first stage model.

        This method is mainly useful for model selection.

        Parameters
        ----------
        W : ndarray
            The `(N, D)` model covariates - which includes the controls *and*
            the treatment variables. The treatment variables should be indexed
            by `treatment_index` passed into this classes' constructor.

        Returns
        -------
        z_hat : ndarray
            The `(N, K)` array of predicted outcomes (from the second stage
            model).
        z : ndarray
            The array of extracted treatment targets.
        """
        check_is_fitted(self, attributes=['alpha_', 'beta_d_'])
        W = check_array(W)
        _, X, z = self._splitW(W)

        # Stage 1
        z_hat = X @ self.beta_c_
        return z_hat, z

    def score_stage1(
        self,
        W: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Get the R^2 score of the first stage regression model.

        This is like the `score` function of scikit learn, but operates only on
        the first stage regression model. That is, it is the score of how well
        the model predicts the *treatments* from the *controls*.

        Parameters
        ----------
        W : ndarray
            The `(N, D)` model covariates - which includes the controls *and*
            the treatment variables. The treatment variables should be indexed
            by `treatment_index` passed into this classes' constructor.
        sample_weight : ndarray
            An array of shape `(N,)` of weights to give each sample in the
            computation of the R^2 score.

        Returns
        -------
        r2 : float
            The R^2 score of the predictions. This is from a call to
            `sklearn.metrics.r2_score`, and so handles multiple outputs in the
            same fashion.
        """
        z_hat, z = self.predict_stage1(W)
        r2: float = r2_score(z, z_hat, sample_weight=sample_weight)
        return r2

    def model_statistics(self) -> StatisticalResults:
        """Return the model statistics.

        This will throw an error if the model has not been fitted.

        Returns
        -------
        stats: StatisticalResults
            The model statistics, including the average treatment effect, its
            standard error, the degrees of freedom used to compute the standard
            error, and a t-statistic and p-value for a two sided t-test to see
            if the average treatment effect is significantly different from
            zero.
        """
        check_is_fitted(self, attributes=['alpha_', 'se_alpha_'])
        stats = StatisticalResults(
            alpha=np.squeeze(self.alpha_),
            std_err=np.squeeze(self.se_alpha_),
            dof=self.dof_,
            t_stat=np.squeeze(self.t_),
            p_value=np.squeeze(self.p_)
        )
        return stats

    def get_params(self, deep: bool = True) -> dict:
        """Get this estimator's initialisation parameters."""
        return {
            'treatment_index': self.treatment_index,
            'regulariser1': self.regulariser1,
            'regulariser2': self.regulariser2,
            'fit_intercept': self.fit_intercept,
        }

    def set_params(self, **parameters: dict) -> Self:
        """Set this estimator's initialisation parameters."""
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
        if gamma.shape != (D,):  # type: ignore
            raise TypeError('gamma has to be a scalar or vector of X.shape[1]')
        gamma_diag = gamma  # type: ignore

    A = X.T @ X + np.diag(gamma_diag)
    b = X.T @ Y
    weights: np.ndarray = solve(A, b, assume_a='pos')
    return weights


#
# Private module utilities
#

@singledispatch
def _check_treatment_index(  # type: ignore
    treatment_index,
    W: np.ndarray,
    fit_intercept: bool
) -> Union[int, np.ndarray, slice]:
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


@_check_treatment_index.register  # type: ignore
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


@_check_treatment_index.register  # type: ignore
def _(treatment_index: np.ndarray, W: np.ndarray, fit_intercept: bool) \
        -> np.ndarray:
    D = W.shape[1]

    adjust_tind = np.copy(treatment_index)
    for n, i in enumerate(treatment_index):
        if (i >= D) or (i < -D):
            raise ValueError(f'treatment_index {i} is out of bounds.')

        # Make sure adjust_tind indexes right weights for initialisation
        if fit_intercept and (i < 0):
            adjust_tind[n] = D + treatment_index[n]

    return adjust_tind
