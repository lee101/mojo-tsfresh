from __future__ import annotations

import numpy as np
import pandas as pd

from . import feature_calculators
from .settings import ComprehensiveFCParameters


def _parameter_name(param):
    def value(v):
        return f'"{v}"' if isinstance(v, str) else str(v)

    return "__".join(f"{key}_{value(param[key])}" for key in sorted(param))


def _calculate(kind, values, settings):
    features = {}
    for name, parameter_list in settings.items():
        if not hasattr(feature_calculators, name):
            raise ValueError(f"Unsupported feature calculator: {name}")
        function = getattr(feature_calculators, name)
        if getattr(function, "fctype", None) == "combiner":
            if not parameter_list:
                continue
            results = function(values, param=parameter_list)
        elif parameter_list:
            results = (
                (_parameter_name(parameters), function(values, **parameters))
                for parameters in parameter_list
            )
        else:
            results = [("", function(values))]
        for key, result in results:
            column = f"{kind}__{name}"
            if key:
                column += f"__{key}"
            features[column] = result
    return features


def extract_features(
    timeseries_container,
    default_fc_parameters=None,
    kind_to_fc_parameters=None,
    column_id=None,
    column_sort=None,
    column_kind=None,
    column_value=None,
    chunksize=None,
    n_jobs=0,
    show_warnings=False,
    disable_progressbar=False,
    impute_function=None,
    profile=False,
    profiling_filename="profile.txt",
    profiling_sorting="cumulative",
    distributor=None,
    pivot=True,
):
    del chunksize, n_jobs, show_warnings, disable_progressbar
    del profile, profiling_filename, profiling_sorting, distributor
    if not isinstance(timeseries_container, pd.DataFrame):
        raise TypeError("timeseries_container must be a pandas DataFrame")
    if column_id is None:
        raise ValueError("column_id must be specified")
    if not pivot:
        raise NotImplementedError("pivot=False is outside the covered subset")

    frame = timeseries_container.copy()
    if column_sort is not None:
        frame = frame.sort_values([column_id, column_sort], kind="mergesort")
    default_settings = (
        ComprehensiveFCParameters()
        if default_fc_parameters is None
        else default_fc_parameters
    )

    excluded = {column_id, column_sort, column_kind}
    rows = {}
    ids = list(pd.unique(frame[column_id]))
    for sample_id in ids:
        sample = frame[frame[column_id] == sample_id]
        combined = {}
        if column_value is not None:
            if column_kind is None:
                groups = [(column_value, sample[column_value])]
            else:
                groups = sample.groupby(column_kind, sort=True)[column_value]
        else:
            value_columns = [column for column in frame.columns if column not in excluded]
            groups = [(column, sample[column]) for column in value_columns]

        for kind, series in groups:
            settings = default_settings
            if kind_to_fc_parameters is not None:
                settings = kind_to_fc_parameters.get(kind, default_settings)
            values = np.ascontiguousarray(series.to_numpy(), dtype=np.float64)
            combined.update(_calculate(kind, values, settings))
        rows[sample_id] = combined

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = None
    if impute_function is not None:
        impute_function(result)
    return result
