"""Test module for model_selection utilities."""

import pytest

from sklearn.metrics import check_scoring
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from twostageridge import TwoStageRidge, first_stage_r2_score
from conftest import make_dag1D_data


@pytest.mark.parametrize('estimator', [
    TwoStageRidge(treatment_index=0),
    make_pipeline(StandardScaler(), TwoStageRidge(treatment_index=0))
])
@pytest.mark.parametrize('scoring', [None, 'r2', first_stage_r2_score])
def test_score_calling(estimator, scoring):
    """Test first_stage_r2_score can be used with TwoStageRidge."""
    W, X, Y, Z = make_dag1D_data()

    estimator.fit(W, Y)
    check_scoring(estimator, scoring=scoring)


def test_first_stage_r2_score_exceptions():
    """Test the correct type exceptions are called in first_stage_r2_score."""
    W, X, Y, Z = make_dag1D_data()

    with pytest.raises(TypeError, match='Estimator .*'):
        first_stage_r2_score(LinearRegression(), W, Y)

    with pytest.raises(TypeError, match='.* last stage .* Pipeline .*'):
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        first_stage_r2_score(pipe, W, Y)
