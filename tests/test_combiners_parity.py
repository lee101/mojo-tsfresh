import numpy as np
import pandas as pd
import pytest

from mojo_tsfresh.feature_extraction import feature_calculators as mojo
from tsfresh.feature_extraction import feature_calculators as upstream


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(17)
    return np.ascontiguousarray(rng.normal(size=400).cumsum() * 0.1)


def assert_combiner_close(actual, expected, tolerance=2e-9):
    actual = list(actual)
    expected = list(expected)
    assert [key for key, _ in actual] == [key for key, _ in expected]
    assert np.allclose(
        [value for _, value in actual],
        [value for _, value in expected],
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )


def test_agg_autocorrelation(signal):
    param = [
        {"f_agg": "mean", "maxlag": 10},
        {"f_agg": "var", "maxlag": 25},
        {"f_agg": "std", "maxlag": 40},
        {"f_agg": "median", "maxlag": 16},
    ]
    assert_combiner_close(
        mojo.agg_autocorrelation(signal, param),
        upstream.agg_autocorrelation(signal, param),
    )


def test_energy_ratio_by_chunks(signal):
    param = [
        {"num_segments": 7, "segment_focus": focus} for focus in range(7)
    ]
    actual = list(mojo.energy_ratio_by_chunks(signal, param))
    expected = list(upstream.energy_ratio_by_chunks(signal, param))
    assert_combiner_close(actual, expected)
    assert sum(value for _, value in actual) == pytest.approx(1.0)


def test_zero_energy_ratio_is_nan():
    x = np.zeros(20)
    param = [{"num_segments": 4, "segment_focus": 2}]
    assert_combiner_close(
        mojo.energy_ratio_by_chunks(x, param),
        upstream.energy_ratio_by_chunks(x, param),
    )


def test_index_mass_quantile(signal):
    param = [{"q": q} for q in (0.1, 0.5, 0.73, 0.95)]
    assert_combiner_close(
        mojo.index_mass_quantile(signal, param),
        upstream.index_mass_quantile(signal, param),
    )


def test_linear_trend(signal):
    param = [
        {"attr": attr}
        for attr in ("pvalue", "rvalue", "intercept", "slope", "stderr")
    ]
    assert_combiner_close(
        mojo.linear_trend(signal, param),
        upstream.linear_trend(signal, param),
    )


def test_aggregated_linear_trend(signal):
    param = [
        {"attr": "slope", "chunk_len": 5, "f_agg": "mean"},
        {"attr": "intercept", "chunk_len": 7, "f_agg": "max"},
        {"attr": "rvalue", "chunk_len": 9, "f_agg": "min"},
        {"attr": "stderr", "chunk_len": 11, "f_agg": "median"},
    ]
    series = pd.Series(signal)
    assert_combiner_close(
        mojo.agg_linear_trend(series, param),
        upstream.agg_linear_trend(series, param),
    )


def test_fft_coefficients(signal):
    param = [
        {"coeff": coefficient, "attr": attr}
        for coefficient in (0, 1, 7, 23, 200, 201)
        for attr in ("real", "imag", "abs", "angle")
    ]
    assert_combiner_close(
        mojo.fft_coefficient(signal, param),
        upstream.fft_coefficient(signal, param),
        tolerance=2e-8,
    )


def test_combiner_feature_names_match_upstream(signal):
    param = [{"attr": "slope"}, {"attr": "pvalue"}]
    assert [x[0] for x in mojo.linear_trend(signal, param)] == [
        x[0] for x in upstream.linear_trend(signal, param)
    ]
