from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_TSFRESH_LIB") or os.path.join(
    ROOT, "dist", "libmojo-tsfresh.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mts_sum_values": ([I, I], F),
    "mts_mean": ([I, I], F),
    "mts_variance": ([I, I], F),
    "mts_standard_deviation": ([I, I], F),
    "mts_abs_energy": ([I, I], F),
    "mts_root_mean_square": ([I, I], F),
    "mts_extreme": ([I, I, I], F),
    "mts_location": ([I, I, I], F),
    "mts_changes": ([I, I, I], F),
    "mts_cid_ce": ([I, I, I], F),
    "mts_count": ([I, I, F, F, I], I),
    "mts_count_mean": ([I, I, I], I),
    "mts_longest_strike": ([I, I, I], I),
    "mts_threshold_statistics": ([I, I, F, I], F),
    "mts_number_crossing_m": ([I, I, F], I),
    "mts_number_peaks": ([I, I, I], I),
    "mts_autocorrelation": ([I, I, I], F),
    "mts_autocorrelations": ([I, I, I, I], None),
    "mts_lag_nonlinearity": ([I, I, I, I], F),
    "mts_binned_entropy": ([I, I, I, I], F),
    "mts_permutation_entropy": ([I, I, I, I, I], F),
    "mts_approximate_entropy": ([I, I, I, F], F),
    "mts_sample_entropy": ([I, I], F),
    "mts_energy_ratio": ([I, I, I, I], F),
    "mts_index_mass_quantile": ([I, I, F], F),
    "mts_linear_trend": ([I, I, I], None),
    "mts_dft_coefficient": ([I, I, I, I, I], None),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first"
            )
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def array(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def addr(x: np.ndarray) -> int:
    return x.ctypes.data
