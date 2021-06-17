"""Test module for model_selection utilities."""
# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.

import pytest

from sklearn.metrics import check_scoring, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from twostageridge import (TwoStageRidge, make_first_stage_scorer,
                           make_combined_stage_scorer)
from conftest import make_dag1D_data


@pytest.mark.parametrize('estimator', [
    TwoStageRidge(treatment_index=0),
    make_pipeline(StandardScaler(), TwoStageRidge(treatment_index=0))
])
@pytest.mark.parametrize('scoring', [None, r2_score, mean_absolute_error])
@pytest.mark.parametrize('score_sign', [False, True, False])
@pytest.mark.parametrize('make_scorer', [
    make_first_stage_scorer,
    make_combined_stage_scorer
])
def test_score_calling(estimator, scoring, score_sign, make_scorer):
    """Test make_first_stage_scorer can be used with TwoStageRidge."""
    W, X, Y, Z = make_dag1D_data()
    estimator.fit(W, Y)

    scorefn = None
    if scoring is not None:
        scorefn = make_scorer(scoring, greater_is_better=score_sign)
    check_scoring(estimator, scoring=scorefn)


@pytest.mark.parametrize('make_scorer', [
    make_first_stage_scorer,
    make_combined_stage_scorer
])
def test_score_exceptions(make_scorer):
    """Test the correct type exceptions are called in the make_*_scorers."""
    W, X, Y, Z = make_dag1D_data()

    scorer = make_scorer(r2_score)

    with pytest.raises(TypeError, match='.* TwoStageRidge estimator.*'):
        scorer(LinearRegression().fit(W, Y), W, Y)

    with pytest.raises(TypeError, match='.* TwoStageRidge estimator.*'):
        pipe = make_pipeline(StandardScaler(), LinearRegression()).fit(W, Y)
        scorer(pipe, W, Y)
