import math

# Based on Graphics Gems


def niceround(num):
    exp = math.floor(math.log10(num))
    f = num / pow(10, exp)  # between 1 and 10
    if f < 1.5:
        nice = 1.
    elif f < 3.0:
        nice = 2.
    elif f < 7.0:
        nice = 5.
    else:
        nice = 10.
    return nice * pow(10, exp)

def niceceil(num):
    exp = math.floor(math.log10(num))
    f = num / pow(10, exp)  # between 1 and 10
    if f < 1.0:
        nice = 1.
    elif f < 2.0:
        nice = 2.
    elif f < 5.0:
        nice = 5.
    else:
        nice = 10.
    return nice * pow(10, exp)

def calc_tick_size(lo, hi, desired_ticks=5):
    nicerange = niceceil(hi - lo)
    delta = niceround(nicerange / (desired_ticks - 1))
    label_min = math.floor(lo / delta) * delta
    label_max = math.ceil(hi / delta) * delta
    return label_min, label_max, delta

def calc_labels(lo, hi, tolerance=.05):
    if lo == hi:
        hi = hi + 1.0
    lmin, lmax, delta = calc_tick_size(lo, hi)
    nfrac = max(-math.floor(math.log10(delta)), 0)  # of fractional digits to show
    label_format = "%%.%df" % nfrac  # simplest axis labels

    num_ticks = int(((lmax - lmin) / delta) + 1)
    # print("lmin=%f lo=%f hi=%f lmax=%f increment=%f num_ticks=%d\n" % (lmin, lo, hi, lmax, delta, num_ticks))

    labels = []
    actual_range = hi - lo
    val = lmin
    fuzzy_tick_tolerance = tolerance * delta
    for i in range(num_ticks):
        # check if tick is within the actual range, but allow a tiny bit of
        # fuzziness due to floating point errors
        if val < lo:
            if abs(val - lo) < fuzzy_tick_tolerance:
                perc = 0.0  # mark it at the bottom
            else:
                perc = None
        elif val > hi:
            if abs(val - hi) < fuzzy_tick_tolerance:
                perc = 100.0  # mark it at the top
            else:
                perc = None
        else:
            perc = (val - lo) / actual_range
        if perc is not None:
            labels.append((perc, label_format % val))
        val += delta
    return labels

def round_minimum_unique_digits(values):
    # Round each entry in the list of numbers to find the minimum number of
    # digits needed to display each value uniquely
    lo = min(values)
    hi = max(values)
    if lo == hi:
        hi += 1.0  # make sure there is some difference
    llo = len(str(abs(int(lo))))
    lhi = len(str(abs(int(hi))))
    #llo, lhi = math.floor(math.log10(lo)), math.floor(math.log10(hi))
    log_range = lhi - llo
    exp_range = math.floor(math.log10(hi - lo))
    base_size = max(llo, lhi)
    stop_at_exp = exp_range - base_size
    #print("lo,hi,range", lo, hi, hi - lo, math.log10(hi - lo), exp_range, llo, lhi, stop_at_exp)
    scale = math.pow(10, stop_at_exp)
    rounded = [int(v / scale) * scale for v in values]
    #print(values)
    #print(rounded)
    fmt = "%%.%df" % abs(stop_at_exp)
    labels = [fmt % r for r in rounded]
    #print(labels)
    return lo, hi, labels


if __name__ == "__main__":
    print((calc_labels(1.2, 27)))
    print((calc_labels(1.2, 2.7)))
    print((round_minimum_unique_digits([142.45, 142.44333, 142.43332, 142.432, 142.4324,12,999])))
    print((round_minimum_unique_digits([142.451109853, 142.44333121983, 142.43332, 142.432, 142.4324])))
    print((round_minimum_unique_digits([142.451109853, 141.44333121983, 140.43332, 139.432, 138.4324])))
    print((round_minimum_unique_digits([3142.451109853, 2141.44333121983, 1140.43332, 139.432, 138.4324])))
    print((round_minimum_unique_digits([0.0, 0.16666666666666666, 0.33333333333333331, 0.5, 0.66666666666666663, 0.83333333333333326, 1.0])))
