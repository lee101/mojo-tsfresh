# mojo-tsfresh

`mojo-tsfresh` is a standalone Mojo port of the compute-heavy core of
[`tsfresh`](https://github.com/blue-yonder/tsfresh), focused on time-series
feature extraction. It exposes the covered calculators under the same names
and signatures as `tsfresh.feature_extraction.feature_calculators`, plus a
compatible pandas `extract_features` path for long and wide tables.

This is an independent package named `mojo_tsfresh`; it does not shadow an
installed `tsfresh`, so the two implementations can be imported together for
parity checks.

## Coverage

There are 59 public calculator names. Forty-five are backed by Mojo kernels:

- Core statistics and changes: energy, sum, mean, variance, standard deviation,
  root mean square, extrema and their locations, absolute and mean changes,
  central second derivative, counts, ranges, threshold ratios, and strikes.
- Time-series structure: `autocorrelation`, `agg_autocorrelation`, `c3`,
  `time_reversal_asymmetry_statistic`, `cid_ce`, crossings, and supported
  peaks.
- Entropy: `binned_entropy`, `permutation_entropy`, `approximate_entropy`, and
  `sample_entropy`.
- Chunk, trend, and frequency features: `energy_ratio_by_chunks`,
  `index_mass_quantile`, `linear_trend`, `agg_linear_trend`, and requested
  `fft_coefficient` bins.

Fourteen low-compute compatibility calculators involving sorting, hashing, or
pandas sample moments use NumPy or pandas: median, quantile, duplicate and
recurrence features, `mean_n_absolute_max`, length, skewness, and kurtosis.

`extract_features` supports a pandas DataFrame in tsfresh's long or wide
layout, `default_fc_parameters`, `kind_to_fc_parameters`, sorting, the standard
pivoted result, and an optional imputation callback. `MinimalFCParameters`,
`EfficientFCParameters`, and `ComprehensiveFCParameters` contain only features
available here.

Not covered are tsfresh's relevance/feature-selection pipeline, Dask and
multiprocessing distributors, `pivot=False`, and specialized calculators that
delegate to larger external algorithms: ADF and AR fits, partial
autocorrelation, CWT/Welch features, matrix profiles, Friedrich coefficients,
change quantiles, Lempel-Ziv complexity, and query-similarity features.

## Install

The repository pins the Mojo nightly dialect used by the source.

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-tsfresh.so`. All project commands run inside
the Pixi environment:

```bash
pixi run test
pixi run bench
```

## Usage

```python
import pandas as pd

from mojo_tsfresh import extract_features

series = pd.DataFrame(
    {
        "id": [1, 1, 1, 1, 2, 2, 2, 2],
        "time": [0, 1, 2, 3, 0, 1, 2, 3],
        "value": [1.0, 2.0, 4.0, 3.0, 2.0, 3.0, 5.0, 8.0],
    }
)

features = extract_features(
    series,
    column_id="id",
    column_sort="time",
    column_value="value",
    default_fc_parameters={
        "mean": None,
        "autocorrelation": [{"lag": 1}],
        "cid_ce": [{"normalize": False}],
    },
)
print(features)
```

Individual calculators retain upstream signatures:

```python
from mojo_tsfresh.feature_extraction import feature_calculators as fc

value = fc.approximate_entropy([1.0, 1.2, 0.9, 1.4, 1.1], m=2, r=0.2)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, and Python 3.13.14. Each entry is the best of five
warm runs against tsfresh 0.21.2 on the same input. A speedup below 1 means
tsfresh was faster.

| Benchmark | mojo-tsfresh | tsfresh 0.21.2 | Speedup |
|---|---:|---:|---:|
| absolute_sum_of_changes (5M) | 5.976 ms | 57.188 ms | 9.57x |
| c3 lag=7 (5M) | 9.052 ms | 81.437 ms | 9.00x |
| number_peaks support=5 (2M) | 41.901 ms | 40.085 ms | 0.96x |
| permutation_entropy d=4 (300k) | 3.540 ms | 535.464 ms | 151.27x |
| approximate_entropy m=2 (1,200) | 10.151 ms | 257.852 ms | 25.40x |
| sample_entropy (2,000) | 32.071 ms | 526.156 ms | 16.41x |
| agg_autocorrelation 200 lags (1,200) | 0.450 ms | 0.624 ms | 1.39x |
| fft_coefficient 4 bins (100k) | 11.516 ms | 1.573 ms | 0.14x |

The entropy and fused window kernels benefit most because Mojo avoids large
temporary arrays and Python-level iteration. Requested Fourier bins use a
direct DFT in this subset; upstream's optimized full real FFT is about 7.3
times faster in the measured four-bin case. Upstream also narrowly wins the
measured support-five peak count.

## How it works

All Mojo kernels live in one compilation unit to keep nightly compiler startup
cost fixed. Python converts inputs to contiguous `float64` arrays. Buffers cross
the C ABI as integer addresses, and the exported Mojo functions reconstruct
`UnsafePointer[Float64, AnyOrigin[mut=True]]` values internally. Results are
returned as scalars or written into NumPy-owned output and scratch buffers, so
the shared library never owns Python memory.

The ctypes layer loads `dist/libmojo-tsfresh.so` and declares every exported
signature explicitly. Calculators fuse loops instead of materializing shifted,
windowed, or pairwise arrays. The extraction layer handles pandas grouping and
upstream-compatible feature-column naming while dispatching numerical work to
those kernels.
