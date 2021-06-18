"""Model selection metrics."""
# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.

import numpy as np
from typing import Optional, Union, Callable, Any
from operator import add

from sklearn.pipeline import Pipeline
from twostageridge import TwoStageRidge


def make_first_stage_scorer(
    score_func: Callable,
    *,
    greater_is_better: bool = True,
    **kwargs: Any
) -> Callable:
    """
    Make a scorer for the first stage of a two stage ridge estimator.

    This can be used for model selection for the `regulariser1` parameter.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        `score_func(y, y_pred, **kwargs)`.
    greater_is_better: bool
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        sign of the outcome of the `score_func` will be flipped.

    **kwargs : dict
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer: callable
        A scorer function that returns a scalar score; greater is better. The
        callable will have the signature `scorer(estimator, X, y,
        sample_weight)` where estimator has to be one of `TwoStageRidge` or
        `Pipeline` with a `TwoStageRidge` final stage.
    """
    scorefn = _partial_scorefn(score_func, greater_is_better, **kwargs)
    scorer = _partial_fs_scorer(scorefn)

    return scorer


def make_combined_stage_scorer(
    score_func: Callable,
    *,
    greater_is_better: bool = True,
    combine_func: Callable = add,
    **kwargs: Any
) -> Callable:
    """
    Make a scorer for both stages of a two stage ridge estimator.

    This can be used for model selection for the `regulariser1` and
    `regulariser2` parameters simultaneously. The scores of both the stages are
    combined together by `combine_func` to produce a single score. By default,
    `combine_func` is addition.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        `score_func(y, y_pred, **kwargs)`.
    greater_is_better: bool
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        sign of the outcome of the `score_func` will be flipped.
    combine_func : callable
        A binary operator that combines both the first and second stage scores.
        This takes the form `func(score1: float, score2: float) -> float`.
    **kwargs : dict
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer: callable
        A scorer function that returns a scalar score; greater is better. The
        callable will have the signature `scorer(estimator, X, y,
        sample_weight)` where estimator has to be one of `TwoStageRidge` or
        `Pipeline` with a `TwoStageRidge` final stage.
    """
    scorefn = _partial_scorefn(score_func, greater_is_better, **kwargs)
    fs_scorer = _partial_fs_scorer(scorefn)

    def scorer(
            estimator: Union[TwoStageRidge, Pipeline],
            W: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None
    ) -> float:
        fs_score = fs_scorer(estimator, W, y, sample_weight)
        y_hat = estimator.predict(W)
        ss_score = score_func(y, y_hat, sample_weight=sample_weight)
        score: float = combine_func(fs_score, ss_score)
        return score

    return scorer


#
# Module partial functions
#

def _partial_scorefn(
        score_func: Callable,
        greater_is_better: bool,
        **kwargs: Any
) -> Callable:
    """Return a partial score function with the sign optionally flipped."""

    def scorefn(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            sample_weight: Optional[np.ndarray]
    ) -> float:
        score: float = score_func(y_true, y_pred, sample_weight=sample_weight,
                                  **kwargs)
        if not greater_is_better:
            score = -score
        return score

    return scorefn


def _partial_fs_scorer(scorefn: Callable) -> Callable:
    """Return a partial scorer function for the first stage model."""

    def scorer(
            estimator: Union[TwoStageRidge, Pipeline],
            W: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Score the first stage model prediction.

        Use this scorer to help with model selection for a TwoStageRidge model.
        In particular, you can use it to choose the `regulariser1` parameter.

        Parameters
        ----------
        estimator : TwoStageRidge or Pipeline
            An instance of the TwoStageRidge estimator, or a Pipeline with the
            last stage being a pipeline.
        W : ndarray
            The `(N, D)` model covariates - which includes the controls *and*
            the treatment variables. The treatment variables should be indexed
            by `treatment_index` passed into this classes' constructor.
        y : ndarray
            The `(N,)` array of outcomes (this is ignored).
        sample_weight : ndarray
            Sample weights.

        Returns
        -------
        score : float
            The score of the predictions. For multiple outputs, see the
            relevant score function documentation as to what is returned.
        """
        if isinstance(estimator, Pipeline):
            features, model = estimator[:-1], estimator[-1]
            Wt = features.transform(W)
        else:
            model = estimator
            Wt = W

        if not hasattr(model, 'predict_stage1'):
            raise TypeError('The estimator or the last stage of the Pipeline '
                            'must be a TwoStageRidge estimator!')

        z_hat, z = model.predict_stage1(Wt)
        score: float = scorefn(z, z_hat, sample_weight)
        return score

    return scorer
