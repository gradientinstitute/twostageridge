"""Model selection metrics."""
# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.

import numpy as np
from typing import Optional, Union

from sklearn.pipeline import Pipeline
from twostageridge import TwoStageRidge


def first_stage_r2_score(
        estimator: Union[Pipeline, TwoStageRidge],
        W: np.ndarray,
        y: Optional[np.ndarray] = None
) -> float:
    """Score the first stage model prediction.

    Use this scorer to help with model selection for a TwoStageRidge model.
    In particular, you can use it to choose the `regulariser1` parameter.

    Parameters
    ----------
    estimator : TwoStageRidge or Pipeline
        An instance of the TwoStageRidge estimator, or a Pipeline with the last
        stage being a pipeline.
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
    if hasattr(estimator, 'score_stage1'):
        r2 = estimator.score_stage1(W)
        return r2

    if isinstance(estimator, Pipeline):
        features, model = estimator[:-1], estimator[-1]
        if not hasattr(model, 'score_stage1'):
            raise TypeError('The last stage of the Pipeline must be a '
                            'TwoStageRidge estimator!')
        r2 = model.score_stage1(features.transform(W))
        return r2

    raise TypeError('Estimator must be an instance of TwoStageRidge or a '
                    'Pipeline!')
