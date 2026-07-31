import numpy as np
import pandas as pd
import pytest

import mojo_tsfresh
import tsfresh
from mojo_tsfresh.feature_extraction.settings import MinimalFCParameters


def data_frame():
    rng = np.random.default_rng(9)
    rows = []
    for sample_id in (10, 20, 30):
        for kind in ("temperature", "pressure"):
            for time in range(60):
                rows.append(
                    {
                        "id": sample_id,
                        "time": time,
                        "kind": kind,
                        "value": rng.normal()
                        + sample_id * 0.01
                        + (0.3 if kind == "pressure" else 0),
                    }
                )
    return pd.DataFrame(rows)


def assert_frame_close(actual, expected):
    assert actual.index.equals(expected.index)
    assert list(actual.columns) == list(expected.columns)
    assert np.allclose(actual, expected, rtol=2e-9, atol=2e-9, equal_nan=True)


def test_minimal_settings_match_upstream():
    frame = data_frame()
    settings = MinimalFCParameters()
    actual = mojo_tsfresh.extract_features(
        frame,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=settings,
    )
    expected = tsfresh.extract_features(
        frame,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=settings,
        n_jobs=0,
        disable_progressbar=True,
    )
    assert_frame_close(actual, expected)


def test_parameterized_long_format_matches_upstream():
    frame = data_frame()
    settings = {
        "autocorrelation": [{"lag": 1}, {"lag": 7}],
        "cid_ce": [{"normalize": False}, {"normalize": True}],
        "number_peaks": [{"n": 2}],
        "permutation_entropy": [{"tau": 1, "dimension": 3}],
        "linear_trend": [{"attr": "slope"}, {"attr": "pvalue"}],
        "energy_ratio_by_chunks": [
            {"num_segments": 4, "segment_focus": focus} for focus in range(4)
        ],
    }
    actual = mojo_tsfresh.extract_features(
        frame,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=settings,
    )
    expected = tsfresh.extract_features(
        frame,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=settings,
        n_jobs=0,
        disable_progressbar=True,
    )
    assert_frame_close(actual, expected)


def test_wide_format_matches_upstream():
    frame = data_frame()
    wide = (
        frame.pivot(index=["id", "time"], columns="kind", values="value")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    settings = {"mean": None, "variance": None, "number_crossing_m": [{"m": 0}]}
    actual = mojo_tsfresh.extract_features(
        wide,
        column_id="id",
        column_sort="time",
        default_fc_parameters=settings,
    )
    expected = tsfresh.extract_features(
        wide,
        column_id="id",
        column_sort="time",
        default_fc_parameters=settings,
        n_jobs=0,
        disable_progressbar=True,
    )
    assert_frame_close(actual, expected)


def test_kind_specific_settings():
    frame = data_frame()
    settings = {
        "temperature": {"mean": None},
        "pressure": {"variance": None},
    }
    actual = mojo_tsfresh.extract_features(
        frame,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters={},
        kind_to_fc_parameters=settings,
    )
    expected = tsfresh.extract_features(
        frame,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters={},
        kind_to_fc_parameters=settings,
        n_jobs=0,
        disable_progressbar=True,
    )
    assert_frame_close(actual, expected)


def test_api_validation():
    with pytest.raises(ValueError, match="column_id"):
        mojo_tsfresh.extract_features(pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(NotImplementedError, match="pivot=False"):
        mojo_tsfresh.extract_features(
            pd.DataFrame({"id": [1], "x": [2]}), column_id="id", pivot=False
        )


def test_calculator_metadata_is_compatible():
    fc = mojo_tsfresh.feature_calculators
    assert fc.mean.fctype == "simple"
    assert fc.mean.minimal is True
    assert fc.fft_coefficient.fctype == "combiner"
    assert fc.sample_entropy.high_comp_cost is True
