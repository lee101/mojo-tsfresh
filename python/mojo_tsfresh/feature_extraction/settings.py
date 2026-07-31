class MinimalFCParameters(dict):
    def __init__(self):
        super().__init__(
            {
                "sum_values": None,
                "median": None,
                "mean": None,
                "length": None,
                "standard_deviation": None,
                "variance": None,
                "root_mean_square": None,
                "maximum": None,
                "absolute_maximum": None,
                "minimum": None,
            }
        )


class ComprehensiveFCParameters(MinimalFCParameters):
    def __init__(self):
        super().__init__()
        self.update(
            {
                "abs_energy": None,
                "absolute_sum_of_changes": None,
                "mean_abs_change": None,
                "mean_change": None,
                "mean_second_derivative_central": None,
                "count_above_mean": None,
                "count_below_mean": None,
                "longest_strike_above_mean": None,
                "longest_strike_below_mean": None,
                "first_location_of_maximum": None,
                "last_location_of_maximum": None,
                "first_location_of_minimum": None,
                "last_location_of_minimum": None,
                "has_duplicate": None,
                "has_duplicate_max": None,
                "has_duplicate_min": None,
                "ratio_value_number_to_time_series_length": None,
                "variation_coefficient": None,
                "variance_larger_than_standard_deviation": None,
                "cid_ce": [{"normalize": True}, {"normalize": False}],
                "autocorrelation": [{"lag": lag} for lag in (1, 2, 3, 5, 10)],
                "c3": [{"lag": lag} for lag in (1, 2, 3)],
                "time_reversal_asymmetry_statistic": [
                    {"lag": lag} for lag in (1, 2, 3)
                ],
                "number_peaks": [{"n": n} for n in (1, 3, 5)],
                "number_crossing_m": [{"m": m} for m in (-1, 0, 1)],
                "binned_entropy": [{"max_bins": 10}],
                "permutation_entropy": [
                    {"tau": 1, "dimension": dimension} for dimension in (3, 4)
                ],
                "linear_trend": [
                    {"attr": attr}
                    for attr in ("pvalue", "rvalue", "intercept", "slope", "stderr")
                ],
                "fft_coefficient": [
                    {"coeff": coeff, "attr": attr}
                    for coeff in (0, 1, 2, 5, 10)
                    for attr in ("real", "imag", "abs", "angle")
                ],
            }
        )


class EfficientFCParameters(ComprehensiveFCParameters):
    pass
