import numpy as np
import pytest

from mojo_tsfresh.feature_extraction import feature_calculators as mojo
from tsfresh.feature_extraction import feature_calculators as upstream


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(2026)
    t = np.arange(320, dtype=np.float64)
    return np.ascontiguousarray(
        0.7 * np.sin(t * 0.071)
        + 0.2 * np.cos(t * 0.013)
        + rng.normal(scale=0.35, size=t.size)
        + 0.001 * t
    )


SIMPLE_CASES = [
    ("sum_values", ()),
    ("mean", ()),
    ("median", ()),
    ("variance", ()),
    ("standard_deviation", ()),
    ("root_mean_square", ()),
    ("abs_energy", ()),
    ("minimum", ()),
    ("maximum", ()),
    ("absolute_maximum", ()),
    ("absolute_sum_of_changes", ()),
    ("mean_abs_change", ()),
    ("mean_change", ()),
    ("mean_second_derivative_central", ()),
    ("cid_ce", (False,)),
    ("cid_ce", (True,)),
    ("first_location_of_maximum", ()),
    ("last_location_of_maximum", ()),
    ("first_location_of_minimum", ()),
    ("last_location_of_minimum", ()),
    ("count_above_mean", ()),
    ("count_below_mean", ()),
    ("longest_strike_above_mean", ()),
    ("longest_strike_below_mean", ()),
    ("range_count", (-0.4, 0.7)),
    ("value_count", (0.0,)),
    ("count_above", (0.2,)),
    ("count_below", (0.2,)),
    ("ratio_beyond_r_sigma", (1.25,)),
    ("large_standard_deviation", (0.15,)),
    ("variance_larger_than_standard_deviation", ()),
    ("variation_coefficient", ()),
    ("number_crossing_m", (0.0,)),
    ("number_peaks", (1,)),
    ("number_peaks", (4,)),
    ("autocorrelation", (1,)),
    ("autocorrelation", (17,)),
    ("c3", (3,)),
    ("time_reversal_asymmetry_statistic", (3,)),
    ("binned_entropy", (10,)),
    ("quantile", (0.23,)),
    ("skewness", ()),
    ("kurtosis", ()),
]


@pytest.mark.parametrize("name,args", SIMPLE_CASES)
def test_simple_calculator_parity(signal, name, args):
    actual = getattr(mojo, name)(signal, *args)
    expected = getattr(upstream, name)(signal, *args)
    assert np.allclose(actual, expected, rtol=2e-9, atol=2e-10, equal_nan=True)


@pytest.mark.parametrize(
    "name,args",
    [
        ("has_duplicate", ()),
        ("has_duplicate_max", ()),
        ("has_duplicate_min", ()),
        ("percentage_of_reoccurring_values_to_all_values", ()),
        ("percentage_of_reoccurring_datapoints_to_all_datapoints", ()),
        ("ratio_value_number_to_time_series_length", ()),
        ("sum_of_reoccurring_values", ()),
        ("sum_of_reoccurring_data_points", ()),
        ("mean_n_absolute_max", (3,)),
        ("length", ()),
    ],
)
def test_discrete_calculator_parity(name, args):
    x = np.array([3.0, -1.0, 3.0, 2.0, 2.0, 2.0, 8.0, -1.0])
    actual = getattr(mojo, name)(x, *args)
    expected = getattr(upstream, name)(x, *args)
    assert actual == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize("tau,dimension", [(1, 3), (2, 4), (3, 5)])
def test_permutation_entropy_parity(signal, tau, dimension):
    assert mojo.permutation_entropy(signal, tau, dimension) == pytest.approx(
        upstream.permutation_entropy(signal, tau, dimension), abs=2e-9
    )


@pytest.mark.parametrize("m,r", [(2, 0.1), (2, 0.2), (3, 0.25)])
def test_approximate_entropy_parity(signal, m, r):
    x = signal[:180]
    assert mojo.approximate_entropy(x, m, r) == pytest.approx(
        upstream.approximate_entropy(x, m, r), abs=2e-8
    )


def test_sample_entropy_parity(signal):
    x = signal[:250]
    assert mojo.sample_entropy(x) == pytest.approx(
        upstream.sample_entropy(x), abs=2e-8
    )


def test_sample_entropy_nan_behavior(signal):
    x = signal.copy()
    x[4] = np.nan
    assert np.isnan(mojo.sample_entropy(x))
    assert np.isnan(upstream.sample_entropy(x))


def test_empty_and_short_series_behavior():
    empty = np.array([], dtype=np.float64)
    single = np.array([3.0])
    assert mojo.sum_values(empty) == upstream.sum_values(empty) == 0
    assert np.isnan(mojo.mean(empty))
    assert np.isnan(mojo.root_mean_square(empty))
    assert np.isnan(mojo.mean_change(single))
    assert mojo.absolute_sum_of_changes(single) == 0
    with pytest.raises(ValueError):
        mojo.minimum(empty)
    with pytest.raises(ValueError):
        upstream.minimum(empty)


def test_approximate_entropy_rejects_negative_r(signal):
    with pytest.raises(ValueError, match="positive"):
        mojo.approximate_entropy(signal, 2, -0.1)
    with pytest.raises(ValueError, match="positive"):
        upstream.approximate_entropy(signal, 2, -0.1)


def test_nan_value_count():
    x = np.array([1.0, np.nan, 2.0, np.nan])
    assert mojo.value_count(x, np.nan) == upstream.value_count(x, np.nan) == 2
