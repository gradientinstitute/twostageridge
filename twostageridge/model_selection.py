"""Model selection utilities."""

import numpy as np
from typing import Optional

from twostageridge import TwoStageRidge


def first_stage_r2_score(
        estimator: TwoStageRidge,
        W: np.ndarray,
        y: Optional[np.ndarray] = None
) -> float:
    """Score the first stage model prediction.

    Use this scorer to help with model selection for a TwoStageRidge model.
    In particular, you can use it to choose the `regulariser1` parameter.

    Parameters
    ----------
    estimator : TwoStageRidge
        An instance of the TwoStageRidge estimator.
    W : ndarray
        The `(N, D)` model covariates - which includes the controls *and* the
        treatment variables. The treatment variables should be indexed by
        `treatment_index` passed into this classes' constructor.
    y : ndarray
        The `(N,)` array of outcomes (this is ignored).

    Returns
    -------
    r2 : float
        The R^2 score of the predictions. This is from a call to
        `sklearn.metrics.r2_score`, and so handles multiple outputs in the same
        fashion.
    """
    if not isinstance(estimator, TwoStageRidge):
        raise TypeError('estimator must be an instance of TwoStageRidge!')

    r2 = estimator.score_stage1(W)
    return r2
