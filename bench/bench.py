from __future__ import annotations

import math
import platform
import time

import numpy as np

from mojo_tsfresh.feature_extraction import feature_calculators as mojo
from tsfresh.feature_extraction import feature_calculators as upstream


def time_best(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in handle
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    return f"{cpu}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main():
    rng = np.random.default_rng(7)
    cases = []

    x_changes = np.ascontiguousarray(rng.normal(size=5_000_000))
    cases.append(
        (
            "absolute_sum_of_changes (5M)",
            lambda: mojo.absolute_sum_of_changes(x_changes),
            lambda: upstream.absolute_sum_of_changes(x_changes),
        )
    )
    cases.append(
        (
            "c3 lag=7 (5M)",
            lambda: mojo.c3(x_changes, 7),
            lambda: upstream.c3(x_changes, 7),
        )
    )

    x_peaks = np.ascontiguousarray(rng.normal(size=2_000_000))
    cases.append(
        (
            "number_peaks support=5 (2M)",
            lambda: mojo.number_peaks(x_peaks, 5),
            lambda: upstream.number_peaks(x_peaks, 5),
        )
    )

    x_permutation = np.ascontiguousarray(rng.normal(size=300_000))
    cases.append(
        (
            "permutation_entropy d=4 (300k)",
            lambda: mojo.permutation_entropy(x_permutation, 1, 4),
            lambda: upstream.permutation_entropy(x_permutation, 1, 4),
        )
    )

    x_approximate = np.ascontiguousarray(rng.normal(size=1_200))
    cases.append(
        (
            "approximate_entropy m=2 (1,200)",
            lambda: mojo.approximate_entropy(x_approximate, 2, 0.2),
            lambda: upstream.approximate_entropy(x_approximate, 2, 0.2),
        )
    )

    x_sample = np.ascontiguousarray(rng.normal(size=2_000))
    cases.append(
        (
            "sample_entropy (2,000)",
            lambda: mojo.sample_entropy(x_sample),
            lambda: upstream.sample_entropy(x_sample),
        )
    )

    x_autocorr = np.ascontiguousarray(rng.normal(size=1_200))
    autocorr_param = [
        {"f_agg": "mean", "maxlag": 200},
        {"f_agg": "var", "maxlag": 200},
    ]
    cases.append(
        (
            "agg_autocorrelation 200 lags (1,200)",
            lambda: mojo.agg_autocorrelation(x_autocorr, autocorr_param),
            lambda: upstream.agg_autocorrelation(x_autocorr, autocorr_param),
        )
    )

    x_fft = np.ascontiguousarray(rng.normal(size=100_000))
    fft_param = [
        {"coeff": coefficient, "attr": "abs"} for coefficient in (1, 2, 5, 10)
    ]
    cases.append(
        (
            "fft_coefficient 4 bins (100k)",
            lambda: list(mojo.fft_coefficient(x_fft, fft_param)),
            lambda: list(upstream.fft_coefficient(x_fft, fft_param)),
        )
    )

    print(f"Machine: {machine()}")
    print()
    print("| Benchmark | mojo-tsfresh | tsfresh 0.21.2 | Speedup |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs in cases:
        ours()
        theirs()
        mojo_time = time_best(ours)
        upstream_time = time_best(theirs)
        speedup = upstream_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
