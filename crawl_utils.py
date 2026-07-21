#!/usr/bin/env python
import math
import random
import io
import itertools as IT
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import crawl_tab as CT


def stepdown(value, step):
    return step * np.log2(1 + value / step)


def stepdown4(value, step, rounding, intmax):

    ret = stepdown(value, step)

    if intmax > 0 and ret > intmax:
        return intmax

    # Randomised rounding
    if rounding == "ROUND_RANDOM":
        intpart = int(ret)
        fracpart = ret - intpart
        if decimal_chance(fracpart):
            intpart += 1
        return intpart

    return ret + (0.5 if rounding == "ROUND_CLOSE" else 0)


def decimal_chance(chance):

    return random.random() < chance


def stepdown_value5(base_value, stepping, first_step, last_step, ceiling_value):

    # Disabling max used to be -1.
    if ceiling_value < 0:
        ceiling_value = 0

    if ceiling_value and ceiling_value < first_step:
        return min(base_value, ceiling_value)

    if base_value < first_step:
        return base_value

    diff = first_step - stepping
    # Since diff < first_step, we can assume here that ceiling_value > diff
    # or ceiling_value == 0.
    return diff + stepdown4(
        base_value - diff,
        stepping,
        "ROUND_DOWN",
        (ceiling_value - diff if ceiling_value else 0),
    )


def check_willpower(wl, power):
    N = 10**6
    adj_pow = int(stepdown(power, 35))
    wlchance = (100 + wl) - adj_pow
    print(f"adj_pow: {adj_pow}")
    print(f"wlchance: {wlchance:.0f}")
    wlch2 = np.random.randint(low=0, high=100, size=N) + np.random.randint(
        low=0, high=101, size=N
    )
    result = wlchance - wlch2
    prob = (result > 0).mean()
    return prob


def hex_success_chance(wl, powc, scale, round_up):
    def _triangular_number(n):
        return n * (n + 1) / 2

    power = int(stepdown(powc, 35))
    target = wl + 100 - power
    denom = 101 * 100
    adjust = denom - 1 if round_up else 0
    if target <= 0:
        return scale
    elif target > 200:
        return 0
    elif target <= 100:
        return (scale * (denom - _triangular_number(target)) + adjust) / denom
    else:
        return (scale * _triangular_number(201 - target) + adjust) / denom


def eval_weapon(
    max_damage,
    prob_to_hit=0.6,
    AC=5,
    delay=1.0,
    N=50000,
    num_attacks=1,
    elec=False,
    holy=False,
    venom=False,
    freezing=False,
    flaming=False,
    draining=False,
    spectral=False,
    num_targets=1,
    gdr=False,
    AC_reduction_multiplier=1,
    label="",
):
    if spectral:
        num_attacks *= 2
    damage = np.random.randint(
        low=1, high=max_damage + 1, size=(num_attacks, N)
    ).astype("float")
    hit_mask = stats.bernoulli.rvs(prob_to_hit, size=(num_attacks, N)).astype(bool)
    damage[~hit_mask] = 0

    if spectral:
        half_hit_mask = hit_mask.copy()
        half_hit_mask[: num_attacks // 2 :, :] = False
        damage[half_hit_mask] *= 0.7

    result = damage
    for i in range(AC_reduction_multiplier):
        AC_reduction = np.random.randint(low=0, high=AC + 1, size=(num_attacks, N))
        if gdr:
            gdr_pct = (16 * AC ** (0.25)) / 100
            min_roll = np.minimum(damage * gdr_pct, AC / 2)
            AC_reduction = np.clip(AC_reduction, min_roll, None)
        result = np.clip(result - AC_reduction, 0, None).astype("float")

    if elec:
        elec_damage = np.random.randint(low=8, high=20 + 1, size=(num_attacks, N))
        elec_mask = stats.bernoulli.rvs(0.25, size=(num_attacks, N)).astype(bool)
        result[elec_mask] += elec_damage[elec_mask]
    if holy:
        result *= 1.75
    if venom:
        result[hit_mask] += 4
    if draining:
        result[hit_mask] *= 1.25
        flat_damage = np.random.randint(low=2, high=4 + 1, size=(num_attacks, N))
        result[hit_mask] += flat_damage[hit_mask]
    if freezing or flaming:
        f_mask = stats.bernoulli.rvs(0.25, size=(num_attacks, N)).astype(bool)
        result[f_mask] *= 1.25

    result = result.sum(axis=0)
    prob_no_damage = (result == 0).astype(int).sum() / len(result)
    prob_whiff = (prob_no_damage) ** num_targets

    expected_damage = result.mean() * (1.0 + (num_targets - 1.0) * 0.7)
    plt.hist(result)
    plt.show()
    if label:
        print(f"""{"-"*40}\n{label}""")
    print(f"expected: {expected_damage}")
    print(f"DPR: {expected_damage / delay}")
    print(f"prob whiff: {prob_whiff:.1%}")


def EHTD(
    attacker,
    defender,
    N=10000,
    num_swings=400,
):
    size = (N, num_swings)
    HP = CT.HP_instances(attacker, defender, size, check_for_death=True)

    # For each row, find the index when the defender has died
    # This is the total number of swings the player makes before dying
    # axis=1 tells argmax to scan across each column
    # Thus total_hits is a 1d-array of length equal to the number of simulations
    total_hits = np.argmax(HP <= 0, axis=1)

    avg_hits = total_hits.mean()
    # ddof=1 calculates the sample var_hits
    std_hits = total_hits.std(ddof=1)
    return avg_hits, std_hits


def div_rand_round(num, den):
    """
    @param num can be an array or number
    @param den is expected to be an int
    Returns an array of the same shape as num.
    Each elements of the array is like a value of div_rand_round(numer, den) from random.cc
    """
    result = num // den
    rem = num % den
    try:
        extra = (np.random.randint(den, size=num.shape) < rem).astype(int)
        mask = rem.astype(bool)
        result[mask] += extra[mask]
    except (AttributeError, TypeError):
        if rem:
            result += int(random.randint(0, den - 1) < rem)
    return result


def random2_div(nom, denom, size):
    if nom <= 0:
        return np.zeros(size, dtype=int)
    else:
        return random2(nom + denom, size) // denom


def random2(maxv, size):
    """
    Returns an array of shape `size`.
    Each elements of the array is like a value of random2(maxv) from random.cc
    The values range from [0, maxv)
    """
    maxv = np.clip(maxv, a_min=1, a_max=None)
    return np.random.randint(0, maxv, size)


def random2avg(maxv, rolls, size):
    """
    Returns an array of shape `size`.
    Each elements of the array is like a value of random2avg(maxv, rolls) from random.cc
    The values range from [0, maxv)
    """
    assert rolls > 1
    try:
        new_size = list(size) + [rolls - 1]
    except TypeError:
        new_size = [size, rolls - 1]
    total = random2(maxv, size) + random2(maxv + 1, new_size).sum(axis=-1)
    return total // rolls


def maybe_random2(x, random_factor, size=1):
    """
    Returns random2(x) if random_factor is true, otherwise the mean.
    [0, x)
    """
    return 0 if (x <= 1) else random2(x, size) if random_factor else x // 2


def maybe_random2_div(nom, denom, random_factor, size=1):
    """
    [0, ceil(nom/denom)]
    """
    return (
        0
        if (nom <= 1)
        else random2(nom + denom, size) // denom
        if random_factor
        else nom // 2 // denom
    )


def fuzz_value(val, lowfuzz, highfuzz, naverage=2, size=1):
    lfuzz = lowfuzz * val // 100
    hfuzz = highfuzz * val // 100
    return val + random2avg(lfuzz + hfuzz + 1, naverage, size) - lfuzz


def roll_dice(num, size, N):
    if num <= 0 or size <= 0:
        return 0
    return np.random.randint(low=1, high=size + 1, size=(N, num)).sum(axis=-1)


def EHTD_table(
    attacker,
    defender,
    AC_range,
    EV_range,
    SH_range,
    N=10000,
    num_swings=200,
):
    data = list()
    for AC, EV, SH in IT.product(AC_range, EV_range, SH_range):
        try:
            defender["AC"] = AC
            defender["EV"] = EV
            defender["SH"] = SH
            av_EHTD, std = EHTD(attacker, defender, N, num_swings)
            data.append([AC, EV, SH, av_EHTD, std])
        except CT.TooFewSwings as err:
            print(err)
            continue
        print(f"{AC}/{EV}/{SH}: {av_EHTD:.1f} ± {std:.1f}")
    df = pd.DataFrame(data, columns=["AC", "EV", "SH", "EHTD", "std"])
    return df


def plot2D_EHTD(
    attacker,
    defender,
    AC_range,
    EV_range,
    SH_range,
    sliceby=["SH", "AC", "EV"],
    N=10000,
    num_swings=200,
    granularity=1,
):

    df = EHTD_table(
        attacker,
        defender,
        AC_range,
        EV_range,
        SH_range,
        N=10000,
        num_swings=200,
    )
    slicecols = dict(
        SH=["AC", "EV", "EHTD"], AC=["EV", "SH", "EHTD"], EV=["AC", "SH", "EHTD"]
    )

    for slice_dir in sliceby:
        for sliceval, dfi in df.groupby(slice_dir):
            title = f"Expected Hits till Death\nAttacker: {describe_attacker(attacker)}\nDefender: {describe_defender(defender, omit=('AC', 'EV', 'SH'))}  {slice_dir} {sliceval:.0f}"
            x, y, z = slicecols[slice_dir]
            CT.make_heatmap(
                dfi[x],
                dfi[y],
                dfi[z],
                title=title,
                xlabel=x,
                ylabel=y,
                granularity=granularity,
            )


def describe_attacker(actor, omit=set()):
    return " ".join(
        [
            f"{key}: {actor[key]}"
            for key in ["damage", "HD", "to_hit", "fighter"]
            if key in actor and key not in omit
        ]
    )


def describe_defender(actor, omit=set()):
    return " ".join(
        [
            f"{key}: {actor[key]}"
            for key in ["HP", "regen", "AC", "EV", "SH"]
            if key in actor and key not in omit
        ]
    )


def plot_differential(
    pairs,
    AC_range,
    EV_range,
    SH_range,
    sliceby="SH",
    N=10000,
    num_swings=200,
    granularity=1,
    xlim=None,
    ylim=None,
):
    """This function may be of dubious value. The EHTD values may have too much
    variance, making this calculation of gaps and shifts show numeric
    peculiarities (contour islands appear where the plot should be smooth).

    Instead of calculating the shift, it might be better to simply use
    CT.plot2D_TAB to get see how EHTD changes as you vary
    2 parameters.

    """
    params = dict(AC_range=AC_range, EV_range=EV_range, SH_range=SH_range)
    AC_dir = ("AC", "AC_range", AC_range)
    EV_dir = ("EV", "EV_range", EV_range)
    SH_dir = ("SH", "SH_range", SH_range)
    slice_dirs = dict(
        SH=[SH_dir, AC_dir, EV_dir],
        AC=[AC_dir, EV_dir, SH_dir],
        EV=[EV_dir, AC_dir, SH_dir],
    )
    zdir, xdir, ydir = slice_dirs[sliceby]
    slice_name, slice_range_name, slice_range = zdir

    for slice_val in slice_range:
        params[slice_range_name] = [slice_val]
        data = list()
        for attacker, defender in pairs:
            defender[slice_name] = slice_val
            df = EHTD_table(attacker, defender, N=N, num_swings=num_swings, **params)
            data.append(df)

        x_name, x_range_name, x_range = xdir
        y_name, y_range_name, y_range = ydir

        xi, yi, XI, YI, ZI_regen = interpolate2d(
            data[1][x_name], data[1][y_name], data[1]["EHTD"], granularity
        )

        xi, yi, XI, YI, ZI = interpolate2d(
            data[0][x_name], data[0][y_name], data[0]["EHTD"], granularity
        )

        regen_worth = list()
        for i, ac in enumerate(xi):
            for j, ev in enumerate(yi):
                ehtd = ZI_regen[i, j]

                ac_gap = ZI[:, j] - ehtd
                i_shift = np.argmax(ac_gap >= 0)
                extra_x_val = xi[i_shift] - xi[i]

                ev_gap = ZI[i, :] - ehtd
                j_shift = np.argmax(ev_gap >= 0)
                extra_y_val = yi[j_shift] - yi[j]

                regen_worth.append((ac, ev, extra_x_val, extra_y_val))
        extra_x = f"extra_{x_name}"
        extra_y = f"extra_{y_name}"
        df = pd.DataFrame(regen_worth, columns=[x_name, y_name, extra_x, extra_y])

        # extra_x or extra_y is < 0 when no index could be found when ZI is bigger than ZI_regen
        df_x = df.loc[df[extra_x] >= 0]
        df_y = df.loc[df[extra_y] >= 0]

        make_heatmap(
            df_x[x_name],
            df_x[y_name],
            df_x[extra_x],
            title=f"{extra_x} ({slice_name}={slice_val})",
            xlabel=x_name,
            ylabel=y_name,
            granularity=granularity,
            xlim=xlim,
            ylim=ylim,
        )
        make_heatmap(
            df_y[x_name],
            df_y[y_name],
            df_y[extra_y],
            title=f"{extra_y} ({slice_name}={slice_val})",
            xlabel=x_name,
            ylabel=y_name,
            granularity=granularity,
            xlim=xlim,
            ylim=ylim,
        )


def markdown_to_df(markdown):
    df = pd.read_table(
        io.StringIO(markdown),
        sep="|",
        skipinitialspace=True,
        header=0,
    )
    df = df.iloc[1:]
    df.columns = df.columns.str.strip()
    for col in df:
        try:
            df[col] = df[col].str.strip()
        except AttributeError:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
