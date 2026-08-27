from std.math import atan2, cos, isnan, log, sin, sqrt
from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of

comptime PI = 3.14159265358979323846264338327950288
comptime W = simd_width_of[DType.float64]()
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def nan() -> Float64:
    var zero = 0.0
    return zero / zero


def total(x: FPtr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += x.load[width=W](i)
        i += W
    var result = acc.reduce_add()
    while i < n:
        result += x[i]
        i += 1
    return result


def energy(x: FPtr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var v = x.load[width=W](i)
        acc += v * v
        i += W
    var result = acc.reduce_add()
    while i < n:
        result += x[i] * x[i]
        i += 1
    return result


def average(x: FPtr, n: Int) -> Float64:
    if n == 0:
        return nan()
    return total(x, n) / Float64(n)


def population_variance(x: FPtr, n: Int) -> Float64:
    if n == 0:
        return nan()
    var mean = average(x, n)
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var d = x.load[width=W](i) - mean
        acc += d * d
        i += W
    var result = acc.reduce_add()
    while i < n:
        var d = x[i] - mean
        result += d * d
        i += 1
    return result / Float64(n)


@export("mts_sum_values")
def mts_sum_values(x_addr: Int, n: Int) abi("C") -> Float64:
    if n <= 0:
        return 0.0
    return total(fp(x_addr), n)


@export("mts_mean")
def mts_mean(x_addr: Int, n: Int) abi("C") -> Float64:
    if n <= 0:
        return nan()
    return average(fp(x_addr), n)


@export("mts_variance")
def mts_variance(x_addr: Int, n: Int) abi("C") -> Float64:
    if n <= 0:
        return nan()
    return population_variance(fp(x_addr), n)


@export("mts_standard_deviation")
def mts_standard_deviation(x_addr: Int, n: Int) abi("C") -> Float64:
    if n <= 0:
        return nan()
    return sqrt(population_variance(fp(x_addr), n))


@export("mts_abs_energy")
def mts_abs_energy(x_addr: Int, n: Int) abi("C") -> Float64:
    if n <= 0:
        return 0.0
    return energy(fp(x_addr), n)


@export("mts_root_mean_square")
def mts_root_mean_square(x_addr: Int, n: Int) abi("C") -> Float64:
    if n <= 0:
        return nan()
    return sqrt(energy(fp(x_addr), n) / Float64(n))


@export("mts_extreme")
def mts_extreme(x_addr: Int, n: Int, mode: Int) abi("C") -> Float64:
    if n <= 0:
        return nan()
    var x = fp(x_addr)
    var result = abs(x[0]) if mode == 2 else x[0]
    if isnan(result):
        return result
    for i in range(1, n):
        var value = abs(x[i]) if mode == 2 else x[i]
        if isnan(value):
            return value
        if (mode == 0 and value < result) or (mode != 0 and value > result):
            result = value
    return result


@export("mts_location")
def mts_location(x_addr: Int, n: Int, mode: Int) abi("C") -> Float64:
    if n <= 0:
        return nan()
    var x = fp(x_addr)
    var best = x[0]
    var index = 0
    for i in range(1, n):
        if (
            (mode < 2 and x[i] > best)
            or (mode >= 2 and x[i] < best)
            or ((mode == 1 or mode == 3) and x[i] == best)
        ):
            best = x[i]
            index = i
    if mode == 1 or mode == 3:
        return Float64(index + 1) / Float64(n)
    return Float64(index) / Float64(n)


@export("mts_changes")
def mts_changes(x_addr: Int, n: Int, mode: Int) abi("C") -> Float64:
    if n <= 1:
        if mode == 0:
            return 0.0
        return nan()
    var x = fp(x_addr)
    if mode == 2:
        return (x[n - 1] - x[0]) / Float64(n - 1)
    if mode == 3:
        return (x[n - 1] - x[n - 2] - x[1] + x[0]) / (2.0 * Float64(n - 2)) if n > 2 else nan()
    var acc = 0.0
    for i in range(1, n):
        acc += abs(x[i] - x[i - 1])
    return acc if mode == 0 else acc / Float64(n - 1)


@export("mts_cid_ce")
def mts_cid_ce(x_addr: Int, n: Int, normalize: Int) abi("C") -> Float64:
    if n <= 1:
        return 0.0
    var x = fp(x_addr)
    var scale = 1.0
    if normalize != 0:
        scale = sqrt(population_variance(x, n))
        if scale == 0.0:
            return 0.0
    var acc = 0.0
    for i in range(1, n):
        var d = (x[i] - x[i - 1]) / scale
        acc += d * d
    return sqrt(acc)


@export("mts_count")
def mts_count(
    x_addr: Int, n: Int, low: Float64, high: Float64, mode: Int
) abi("C") -> Int:
    var x = fp(x_addr)
    var count = 0
    for i in range(n):
        if (
            (mode == 0 and x[i] >= low and x[i] < high)
            or (mode == 1 and x[i] == low)
            or (mode == 2 and isnan(x[i]))
            or (mode == 3 and x[i] >= low)
            or (mode == 4 and x[i] <= low)
        ):
            count += 1
    return count


@export("mts_count_mean")
def mts_count_mean(x_addr: Int, n: Int, above: Int) abi("C") -> Int:
    if n <= 0:
        return 0
    var x = fp(x_addr)
    var mean = average(x, n)
    var count = 0
    for i in range(n):
        if (above != 0 and x[i] > mean) or (above == 0 and x[i] < mean):
            count += 1
    return count


@export("mts_longest_strike")
def mts_longest_strike(x_addr: Int, n: Int, above: Int) abi("C") -> Int:
    if n <= 0:
        return 0
    var x = fp(x_addr)
    var mean = average(x, n)
    var longest = 0
    var current = 0
    for i in range(n):
        if (above != 0 and x[i] > mean) or (above == 0 and x[i] < mean):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


@export("mts_threshold_statistics")
def mts_threshold_statistics(
    x_addr: Int, n: Int, r: Float64, mode: Int
) abi("C") -> Float64:
    if n <= 0:
        return nan()
    var x = fp(x_addr)
    var mean = average(x, n)
    var std = sqrt(population_variance(x, n))
    if mode == 0:
        var count = 0
        for i in range(n):
            if abs(x[i] - mean) > r * std:
                count += 1
        return Float64(count) / Float64(n)
    var lo = x[0]
    var hi = x[0]
    for i in range(1, n):
        lo = min(lo, x[i])
        hi = max(hi, x[i])
    return 1.0 if std > r * (hi - lo) else 0.0


@export("mts_number_crossing_m")
def mts_number_crossing_m(x_addr: Int, n: Int, threshold: Float64) abi("C") -> Int:
    if n <= 1:
        return 0
    var x = fp(x_addr)
    var count = 0
    for i in range(1, n):
        if (x[i - 1] > threshold) != (x[i] > threshold):
            count += 1
    return count


def count_peaks(x: FPtr, start: Int, end: Int, support: Int) -> Int:
    var count = SIMD[DType.int64, W](0)
    var i = start
    while i + W <= end:
        var center = x.load[width=W](i)
        var peaks = SIMD[DType.bool, W](fill=True)
        for offset in range(1, support + 1):
            peaks &= center.gt(x.load[width=W](i - offset))
            peaks &= center.gt(x.load[width=W](i + offset))
        count += peaks.cast[DType.int64]()
        i += W
    var result = Int(count.reduce_add())
    while i < end:
        var peak = True
        for offset in range(1, support + 1):
            if not (x[i] > x[i - offset] and x[i] > x[i + offset]):
                peak = False
        if peak:
            result += 1
        i += 1
    return result


@export("mts_number_peaks")
def mts_number_peaks(x_addr: Int, length: Int, support: Int) abi("C") -> Int:
    if support <= 0 or 2 * support >= length:
        return 0
    return count_peaks(fp(x_addr), support, length - support, support)


@export("mts_number_peaks_parallel")
def mts_number_peaks_parallel(
    x_addr: Int, length: Int, support: Int, scratch_addr: Int, workers: Int
) abi("C") -> Int:
    if support <= 0 or 2 * support >= length:
        return 0
    var x = fp(x_addr)
    var scratch = ip(scratch_addr)
    var candidates = length - 2 * support

    @__parameter
    def work(worker: Int):
        var start = support + candidates * worker // workers
        var end = support + candidates * (worker + 1) // workers
        scratch[worker] = Int64(count_peaks(x, start, end, support))

    initialize_runtime()
    parallelize[work](workers, workers)
    var result = 0
    for worker in range(workers):
        result += Int(scratch[worker])
    return result


@export("mts_autocorrelation")
def mts_autocorrelation(x_addr: Int, n: Int, lag: Int) abi("C") -> Float64:
    if lag < 0 or n < lag or n <= 0:
        return nan()
    var x = fp(x_addr)
    var mean = average(x, n)
    var variance = population_variance(x, n)
    if abs(variance) <= 1.0e-8:
        return nan()
    var acc = 0.0
    for i in range(n - lag):
        acc += (x[i] - mean) * (x[i + lag] - mean)
    return acc / (Float64(n - lag) * variance)


@export("mts_autocorrelations")
def mts_autocorrelations(
    x_addr: Int, dst_addr: Int, n: Int, max_lag: Int
) abi("C"):
    var x = fp(x_addr)
    var dst = fp(dst_addr)
    var mean = average(x, n)
    var variance = population_variance(x, n)
    for lag in range(1, max_lag + 1):
        if lag >= n or abs(variance) < 1.0e-10:
            dst[lag - 1] = 0.0
        else:
            var acc = 0.0
            for i in range(n - lag):
                acc += (x[i] - mean) * (x[i + lag] - mean)
            dst[lag - 1] = acc / (Float64(n - lag) * variance)


@export("mts_lag_nonlinearity")
def mts_lag_nonlinearity(
    x_addr: Int, n: Int, lag: Int, mode: Int
) abi("C") -> Float64:
    if lag < 0 or 2 * lag >= n:
        return 0.0
    var x = fp(x_addr)
    var acc = 0.0
    for i in range(n - 2 * lag):
        var a = x[i]
        var b = x[i + lag]
        var c = x[i + 2 * lag]
        if mode == 0:
            acc += a * b * c
        else:
            acc += c * c * b - b * a * a
    return acc / Float64(n - 2 * lag)


@export("mts_binned_entropy")
def mts_binned_entropy(
    x_addr: Int, counts_addr: Int, n: Int, bins: Int
) abi("C") -> Float64:
    if n <= 0 or bins <= 0:
        return nan()
    var x = fp(x_addr)
    var counts = ip(counts_addr)
    var lo = x[0]
    var hi = x[0]
    for i in range(n):
        if isnan(x[i]):
            return nan()
        lo = min(lo, x[i])
        hi = max(hi, x[i])
    for j in range(bins):
        counts[j] = 0
    if hi == lo:
        return 0.0
    var scale = Float64(bins) / (hi - lo)
    for i in range(n):
        var index = Int((x[i] - lo) * scale)
        if index == bins:
            index -= 1
        counts[index] += 1
    var result = 0.0
    for j in range(bins):
        if counts[j] > 0:
            var p = Float64(counts[j]) / Float64(n)
            result -= p * log(p)
    return result


def factorial(n: Int) -> Int:
    var result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@export("mts_permutation_entropy")
def mts_permutation_entropy(
    x_addr: Int,
    counts_addr: Int,
    n: Int,
    tau: Int,
    dimension: Int,
) abi("C") -> Float64:
    var windows = (n - dimension) // tau + 1
    if windows <= 0:
        return nan()
    var patterns = factorial(dimension)
    var x = fp(x_addr)
    var counts = ip(counts_addr)
    for i in range(patterns):
        counts[i] = 0
    for window in range(windows):
        var start = window * tau
        var code = 0
        for j in range(dimension):
            var digit = 0
            for k in range(j + 1, dimension):
                if x[start + k] < x[start + j]:
                    digit += 1
            code = code * (dimension - j) + digit
        counts[code] += 1
    var result = 0.0
    for i in range(patterns):
        if counts[i] > 0:
            var p = Float64(counts[i]) / Float64(windows)
            result -= p * log(p)
    return result


def phi_entropy(x: FPtr, n: Int, m: Int, tolerance: Float64) -> Float64:
    var templates = n - m + 1
    var result = 0.0
    for i in range(templates):
        var matches = 0
        for j in range(templates):
            var close = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > tolerance:
                    close = False
            if close:
                matches += 1
        result += log(Float64(matches) / Float64(templates))
    return result / Float64(templates)


@export("mts_approximate_entropy")
def mts_approximate_entropy(
    x_addr: Int, n: Int, m: Int, r: Float64
) abi("C") -> Float64:
    if r < 0.0:
        return nan()
    if n <= m + 1:
        return 0.0
    var x = fp(x_addr)
    var tolerance = r * sqrt(population_variance(x, n))
    return abs(phi_entropy(x, n, m, tolerance) - phi_entropy(x, n, m + 1, tolerance))


def template_pair_count(x: FPtr, n: Int, m: Int, tolerance: Float64) -> Int:
    var templates = n - m + 1
    var count = 0
    for i in range(templates):
        for j in range(templates):
            if i == j:
                continue
            var close = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > tolerance:
                    close = False
            if close:
                count += 1
    return count


@export("mts_sample_entropy")
def mts_sample_entropy(x_addr: Int, n: Int) abi("C") -> Float64:
    if n < 3:
        return nan()
    var x = fp(x_addr)
    for i in range(n):
        if isnan(x[i]):
            return nan()
    var tolerance = 0.2 * sqrt(population_variance(x, n))
    var b = template_pair_count(x, n, 2, tolerance)
    var a = template_pair_count(x, n, 3, tolerance)
    return -log(Float64(a) / Float64(b))


@export("mts_energy_ratio")
def mts_energy_ratio(
    x_addr: Int, n: Int, segments: Int, focus: Int
) abi("C") -> Float64:
    var x = fp(x_addr)
    var full = energy(x, n)
    if full == 0.0:
        return nan()
    var base = n // segments
    var remainder = n % segments
    var start = focus * base + min(focus, remainder)
    var size = base + (1 if focus < remainder else 0)
    return energy(x + start, size) / full


@export("mts_index_mass_quantile")
def mts_index_mass_quantile(x_addr: Int, n: Int, q: Float64) abi("C") -> Float64:
    var x = fp(x_addr)
    var mass = 0.0
    for i in range(n):
        mass += abs(x[i])
    if mass == 0.0:
        return nan()
    var cumulative = 0.0
    for i in range(n):
        cumulative += abs(x[i])
        if cumulative / mass >= q:
            return Float64(i + 1) / Float64(n)
    return 1.0


@export("mts_linear_trend")
def mts_linear_trend(x_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var x = fp(x_addr)
    var dst = fp(dst_addr)
    if n < 2:
        for i in range(5):
            dst[i] = nan()
        return
    var sx = Float64(n * (n - 1)) / 2.0
    var sxx = Float64(n * (n - 1) * (2 * n - 1)) / 6.0
    var sy = total(x, n)
    var syy = energy(x, n)
    var sxy = 0.0
    for i in range(n):
        sxy += Float64(i) * x[i]
    var denom_x = Float64(n) * sxx - sx * sx
    var denom_y = Float64(n) * syy - sy * sy
    var slope = (Float64(n) * sxy - sx * sy) / denom_x
    var intercept = (sy - slope * sx) / Float64(n)
    var rvalue = (Float64(n) * sxy - sx * sy) / sqrt(denom_x * denom_y) if denom_y > 0.0 else 0.0
    var stderr = nan()
    if n > 2 and denom_y > 0.0:
        stderr = sqrt((1.0 - rvalue * rvalue) * denom_y / denom_x / Float64(n - 2))
    dst[0] = slope
    dst[1] = intercept
    dst[2] = rvalue
    dst[3] = stderr
    dst[4] = Float64(n)


@export("mts_dft_coefficient")
def mts_dft_coefficient(
    x_addr: Int, n: Int, coefficient: Int, real_addr: Int, imag_addr: Int
) abi("C"):
    var x = fp(x_addr)
    var real_dst = fp(real_addr)
    var imag_dst = fp(imag_addr)
    if coefficient < 0 or coefficient > n // 2:
        real_dst[0] = nan()
        imag_dst[0] = nan()
        return
    var step = -2.0 * PI * Float64(coefficient) / Float64(n)
    var re = 0.0
    var im = 0.0
    for i in range(n):
        var angle = step * Float64(i)
        re += x[i] * cos(angle)
        im += x[i] * sin(angle)
    real_dst[0] = re
    imag_dst[0] = im
