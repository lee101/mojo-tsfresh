from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.special import stdtr

from .._lib import addr, array, lib


_PEAKS_PARALLEL_THRESHOLD = 262_144
_PEAKS_WORKERS = 8


def set_property(key, value):
    def decorate_func(func):
        setattr(func, key, value)
        return func

    return decorate_func


def _simple(func):
    return set_property("fctype", "simple")(func)


def _combiner(func):
    return set_property("fctype", "combiner")(func)


def _x(x) -> np.ndarray:
    return array(x)


@set_property("minimal", True)
@_simple
def sum_values(x):
    a = _x(x)
    return lib().mts_sum_values(addr(a), a.size)


@set_property("minimal", True)
@_simple
def mean(x):
    a = _x(x)
    return lib().mts_mean(addr(a), a.size)


@set_property("minimal", True)
@_simple
def variance(x):
    a = _x(x)
    return lib().mts_variance(addr(a), a.size)


@set_property("minimal", True)
@_simple
def standard_deviation(x):
    a = _x(x)
    return lib().mts_standard_deviation(addr(a), a.size)


@set_property("minimal", True)
@_simple
def root_mean_square(x):
    a = _x(x)
    return lib().mts_root_mean_square(addr(a), a.size)


@_simple
def abs_energy(x):
    a = _x(x)
    return lib().mts_abs_energy(addr(a), a.size)


@set_property("minimal", True)
@_simple
def minimum(x):
    a = _x(x)
    if not a.size:
        raise ValueError("zero-size array to reduction operation minimum which has no identity")
    return lib().mts_extreme(addr(a), a.size, 0)


@set_property("minimal", True)
@_simple
def maximum(x):
    a = _x(x)
    if not a.size:
        raise ValueError("zero-size array to reduction operation maximum which has no identity")
    return lib().mts_extreme(addr(a), a.size, 1)


@set_property("minimal", True)
@_simple
def absolute_maximum(x):
    a = _x(x)
    return lib().mts_extreme(addr(a), a.size, 2)


@set_property("minimal", True)
@_simple
def median(x):
    return np.median(np.asarray(x))


@_simple
def quantile(x, q):
    return np.quantile(np.asarray(x), q) if len(x) else np.nan


@_simple
def absolute_sum_of_changes(x):
    a = _x(x)
    return lib().mts_changes(addr(a), a.size, 0)


@_simple
def mean_abs_change(x):
    a = _x(x)
    return lib().mts_changes(addr(a), a.size, 1)


@_simple
def mean_change(x):
    a = _x(x)
    return lib().mts_changes(addr(a), a.size, 2)


@_simple
def mean_second_derivative_central(x):
    a = _x(x)
    return lib().mts_changes(addr(a), a.size, 3)


@_simple
def cid_ce(x, normalize):
    a = _x(x)
    return lib().mts_cid_ce(addr(a), a.size, int(normalize))


@_simple
def first_location_of_maximum(x):
    a = _x(x)
    return lib().mts_location(addr(a), a.size, 0)


@_simple
def last_location_of_maximum(x):
    a = _x(x)
    return lib().mts_location(addr(a), a.size, 1)


@_simple
def first_location_of_minimum(x):
    a = _x(x)
    return lib().mts_location(addr(a), a.size, 2)


@_simple
def last_location_of_minimum(x):
    a = _x(x)
    return lib().mts_location(addr(a), a.size, 3)


@_simple
def count_above_mean(x):
    a = _x(x)
    return lib().mts_count_mean(addr(a), a.size, 1)


@_simple
def count_below_mean(x):
    a = _x(x)
    return lib().mts_count_mean(addr(a), a.size, 0)


@_simple
def longest_strike_above_mean(x):
    a = _x(x)
    return lib().mts_longest_strike(addr(a), a.size, 1)


@_simple
def longest_strike_below_mean(x):
    a = _x(x)
    return lib().mts_longest_strike(addr(a), a.size, 0)


@_simple
def range_count(x, min, max):
    a = _x(x)
    return lib().mts_count(addr(a), a.size, float(min), float(max), 0)


@_simple
def value_count(x, value):
    a = _x(x)
    mode = 2 if np.isnan(value) else 1
    return lib().mts_count(addr(a), a.size, float(value), 0.0, mode)


@_simple
def count_above(x, t):
    a = _x(x)
    if not a.size:
        return np.nan
    return lib().mts_count(addr(a), a.size, float(t), 0.0, 3) / len(a)


@_simple
def count_below(x, t):
    a = _x(x)
    if not a.size:
        return np.nan
    return lib().mts_count(addr(a), a.size, float(t), 0.0, 4) / len(a)


@_simple
def ratio_beyond_r_sigma(x, r):
    a = _x(x)
    return lib().mts_threshold_statistics(addr(a), a.size, float(r), 0)


@_simple
def large_standard_deviation(x, r):
    a = _x(x)
    return bool(lib().mts_threshold_statistics(addr(a), a.size, float(r), 1))


@_simple
def variance_larger_than_standard_deviation(x):
    v = variance(x)
    return bool(v > math.sqrt(v))


@_simple
def variation_coefficient(x):
    avg = mean(x)
    return np.nan if avg == 0 else standard_deviation(x) / avg


@_simple
def number_crossing_m(x, m):
    a = _x(x)
    return lib().mts_number_crossing_m(addr(a), a.size, float(m))


@_simple
def number_peaks(x, n):
    a = _x(x)
    if a.size >= _PEAKS_PARALLEL_THRESHOLD:
        scratch = np.empty(_PEAKS_WORKERS, dtype=np.int64)
        return lib().mts_number_peaks_parallel(
            addr(a), a.size, int(n), addr(scratch), _PEAKS_WORKERS
        )
    return lib().mts_number_peaks(addr(a), a.size, int(n))


@_simple
def autocorrelation(x, lag):
    a = _x(x)
    return lib().mts_autocorrelation(addr(a), a.size, int(lag))


@_combiner
def agg_autocorrelation(x, param):
    a = _x(x)
    max_lag = max(config["maxlag"] for config in param)
    values = np.empty(max_lag, dtype=np.float64)
    lib().mts_autocorrelations(addr(a), addr(values), a.size, max_lag)
    return [
        (
            f'f_agg_"{config["f_agg"]}"__maxlag_{config["maxlag"]}',
            getattr(np, config["f_agg"])(values[: int(config["maxlag"])]),
        )
        for config in param
    ]


@_simple
def c3(x, lag):
    a = _x(x)
    return lib().mts_lag_nonlinearity(addr(a), a.size, int(lag), 0)


@_simple
def time_reversal_asymmetry_statistic(x, lag):
    a = _x(x)
    return lib().mts_lag_nonlinearity(addr(a), a.size, int(lag), 1)


@_simple
def binned_entropy(x, max_bins):
    a = _x(x)
    counts = np.empty(max_bins, dtype=np.int64)
    return lib().mts_binned_entropy(addr(a), addr(counts), a.size, int(max_bins))


@_simple
def permutation_entropy(x, tau, dimension):
    assert dimension > 1
    assert tau > 0
    a = _x(x)
    counts = np.empty(math.factorial(dimension), dtype=np.int64)
    return lib().mts_permutation_entropy(
        addr(a), addr(counts), a.size, int(tau), int(dimension)
    )


@set_property("high_comp_cost", True)
@_simple
def approximate_entropy(x, m, r):
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    a = _x(x)
    return lib().mts_approximate_entropy(addr(a), a.size, int(m), float(r))


@set_property("high_comp_cost", True)
@_simple
def sample_entropy(x):
    a = _x(x)
    return lib().mts_sample_entropy(addr(a), a.size)


@_combiner
def energy_ratio_by_chunks(x, param):
    a = _x(x)
    result = []
    for config in param:
        segments = int(config["num_segments"])
        focus = int(config["segment_focus"])
        assert focus < segments
        assert segments > 0
        key = f"num_segments_{segments}__segment_focus_{focus}"
        value = lib().mts_energy_ratio(addr(a), a.size, segments, focus)
        result.append((key, value))
    return result


@_combiner
def index_mass_quantile(x, param):
    a = _x(x)
    return [
        (
            f"q_{config['q']}",
            lib().mts_index_mass_quantile(addr(a), a.size, float(config["q"])),
        )
        for config in param
    ]


def _linregress_mojo(x):
    a = _x(x)
    stats = np.empty(5, dtype=np.float64)
    lib().mts_linear_trend(addr(a), addr(stats), a.size)
    slope, intercept, rvalue, stderr, n = stats
    if n > 2:
        tiny = 1.0e-20
        t_stat = rvalue * math.sqrt((n - 2) / ((1.0 - rvalue + tiny) * (1.0 + rvalue + tiny)))
        pvalue = 2.0 * stdtr(n - 2, -abs(t_stat))
    else:
        pvalue = 0.0 if abs(rvalue) == 1.0 else 1.0
    return {
        "slope": slope,
        "intercept": intercept,
        "rvalue": rvalue,
        "pvalue": pvalue,
        "stderr": stderr,
    }


@_combiner
def linear_trend(x, param):
    result = _linregress_mojo(x)
    return [
        (f'attr_"{config["attr"]}"', result[config["attr"]])
        for config in param
    ]


def _aggregate_on_chunks(x, f_agg, chunk_len):
    return [
        getattr(np, f_agg)(x[i * chunk_len : (i + 1) * chunk_len])
        for i in range(int(np.ceil(len(x) / chunk_len)))
    ]


@_combiner
def agg_linear_trend(x, param):
    series = np.asarray(x)
    cache = defaultdict(dict)
    values = []
    keys = []
    for config in param:
        chunk_len = config["chunk_len"]
        f_agg = config["f_agg"]
        attr = config["attr"]
        if chunk_len >= len(series):
            value = np.nan
        else:
            if chunk_len not in cache[f_agg]:
                aggregated = _aggregate_on_chunks(series, f_agg, chunk_len)
                cache[f_agg][chunk_len] = _linregress_mojo(aggregated)
            value = cache[f_agg][chunk_len][attr]
        keys.append(f'attr_"{attr}"__chunk_len_{chunk_len}__f_agg_"{f_agg}"')
        values.append(value)
    return zip(keys, values)


@_combiner
def fft_coefficient(x, param):
    assert min(config["coeff"] for config in param) >= 0
    assert {config["attr"] for config in param} <= {"imag", "real", "abs", "angle"}
    a = _x(x)
    transformed = np.fft.rfft(a)
    values = []
    keys = []
    for config in param:
        coefficient = int(config["coeff"])
        attr = config["attr"]
        if coefficient >= transformed.size:
            value = np.nan
        else:
            z = transformed[coefficient]
            if attr == "real":
                value = z.real
            elif attr == "imag":
                value = z.imag
            elif attr == "abs":
                value = abs(z)
            else:
                value = math.degrees(math.atan2(z.imag, z.real))
        keys.append(f'attr_"{attr}"__coeff_{coefficient}')
        values.append(value)
    return zip(keys, values)


@_simple
def has_duplicate(x):
    a = np.asarray(x)
    return a.size != np.unique(a).size


@_simple
def has_duplicate_max(x):
    a = np.asarray(x)
    return np.sum(a == np.max(a)) >= 2


@_simple
def has_duplicate_min(x):
    a = np.asarray(x)
    return np.sum(a == np.min(a)) >= 2


@_simple
def percentage_of_reoccurring_values_to_all_values(x):
    if len(x) == 0:
        return np.nan
    _, counts = np.unique(x, return_counts=True)
    return np.sum(counts > 1) / float(counts.size) if counts.size else 0.0


@_simple
def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    if len(x) == 0:
        return np.nan
    _, counts = np.unique(x, return_counts=True)
    return counts[counts > 1].sum() / len(x)


@_simple
def ratio_value_number_to_time_series_length(x):
    return len(np.unique(x)) / len(x) if len(x) else np.nan


@_simple
def sum_of_reoccurring_values(x):
    values, counts = np.unique(x, return_counts=True)
    return np.sum(values[counts > 1])


@_simple
def sum_of_reoccurring_data_points(x):
    values, counts = np.unique(x, return_counts=True)
    return np.sum(values[counts > 1] * counts[counts > 1])


@_simple
def mean_n_absolute_max(x, number_of_maxima):
    assert number_of_maxima > 0
    a = np.asarray(x)
    return (
        np.mean(np.sort(np.abs(a))[-number_of_maxima:])
        if len(a) > number_of_maxima
        else np.nan
    )


@_simple
def length(x):
    return len(x)


@_simple
def skewness(x):
    import pandas as pd

    return pd.Series(x).skew(skipna=False)


@_simple
def kurtosis(x):
    import pandas as pd

    return pd.Series(x).kurtosis()


__all__ = [
    name
    for name, value in list(globals().items())
    if callable(value) and hasattr(value, "fctype")
]
