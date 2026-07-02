#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.stats as stats
import crawl_skills as CS
import crawl_utils as CU

MAX_SKILL_LEVEL = 27
MIN_HIT_MISS_PERCENTAGE = 5
PARM_EVASION = dict(none=0, robe=0, steam=0, leather=-40, acid=-50, ring=-70, swamp=-70,
                    quicksilver=-70, scale=-100, pearl=-110, chain=-140,
                    shadow=-150, storm=-150, plate=-180, crystal_plate=-230, golden=-230,
                    cloak=0, gloves=0, scarf=0, helmet=0, hat=0, boots=0, barding=-60,
                    buckler=-50, kite=-100, tower=-150)

PARM_AC = dict(none=0, robe=2, steam=5, leather=3, acid=6, ring=5, swamp=7,
                    quicksilver=9, scale=6, pearl=10, chain=8,
                    shadow=11, storm=10, plate=10, crystal_plate=14, golden=12,
                    cloak=1, gloves=1, scarf=0, helmet=1, hat=0, boots=1, barding=4,
                    buckler=3, kite=8, tower=13)

PWPN_DAMAGE = dict(war_axe=11, battleaxe=15, broad_axe=13, executioners_axe=18, hand_axe=7,
                  spear=6, partisan=14, trident=9, demon_trident=12, bardiche=18, glaive=15, trishula=13, halberd=13,
                  dagger=4, short_sword=5, quick_blade=4, rapier=7, athame=7,
                  demon_blade=13, falchion=8, long_sword=10, scimitar=12, triple_sword=19, great_sword=17, double_sword=15, eudemon_blade=14,
                  mace=8, flail=10, morningstar=13, eveningstar=15, demon_whip=11, giant_spiked_club=22, giant_club=20, great_mace=17, dire_flail=13, sacred_scourge=12, club=5,
                  sling=7, shortbow=8, orcbow=11, longbow=14, arbalest=16, triple_crossbow=23, hand_cannon=16,
                  staff=5, quarterstaff=10, lajatang=16
                  )

PWPN_SPEED = dict(war_axe=15, battleaxe=17, broad_axe=16, executioners_axe=19, hand_axe=13,
                  spear=11, partisan=17, trident=13, demon_trident=13, bardiche=19, glaive=17, trishula=13, halberd=15, 
                  dagger=10, short_sword=10, quick_blade=15, rapier=12, athame=13,
                  demon_blade=13, falchion=13, long_sword=14, scimitar=14, triple_sword=18, great_sword=17, double_sword=15, eudemon_blade=12,
                  mace=14, flail=14, morningstar=15, eveningstar=15, demon_whip=11, giant_spiked_club=18, giant_club=16, great_mace=17, dire_flail=13, sacred_scourge=11, club=13,
                  sling=14, shortbow=14, orcbow=15, longbow=17, arbalest=19, triple_crossbow=23, hand_cannon=19,
                  staff=12, quarterstaff=13, lajatang=14
                  )

WEAPON_TO_HIT_BONUS = dict(war_axe=0, battleaxe=-4, broad_axe=-2, executioners_axe=-6, hand_axe=3,
                  spear=4, partisan=1, trident=1, demon_trident=1, bardiche=-6, glaive=-3, trishula=0, halberd=-3,
                  dagger=6, short_sword=4, quick_blade=6, rapier=4, athame=5,
                  demon_blade=-1, falchion=2, long_sword=1, scimitar=0, triple_sword=-4, great_sword=-3, double_sword=-1, eudemon_blade=-2,
                           mace=3, flail=0, morningstar=-2, eveningstar=-1, demon_whip=1, giant_spiked_club=-7, giant_club=-6, great_mace=-4, dire_flail=-3, sacred_scourge=0, club=3,
                  sling=0, shortbow=2, orcbow=-3, longbow=0, arbalest=-2, triple_crossbow=-2, hand_cannon=3,
                  staff=5, quarterstaff=3, lajatang=-3
                  )

class TooFewSwings(Exception):
    pass



def plot1D_TAB(
        combat_gen,
        N=10000,
        num_swings=200,
        granularity=1,
        title="",
        save_as="",
        x=None,
        extra_func=None,
):

    df = TAB_table(
        combat_gen,
        N=N,
        num_swings=num_swings,
    )

    x2 = df.columns.difference(["EKTD", "std_kills", "ETTD", "std_turns", "HD"])
    x = x or x2

    full_title = f"Expected Kills till Death"
    full_save_as = save_as
    if title:
        full_title = f"{full_title}\n{title}"
    if full_save_as:
        full_save_as = f"kills-{save_as}.jpg"

    for HD, grp in df.groupby("HD"):
        plt.plot(grp[x], grp["EKTD"], label=f"HD={HD}")

    plt.xlabel(x)
    plt.ylabel("EKTD")
    plt.title(f"{full_title}")
    plt.legend()

    print(df)
    if full_save_as:
        plt.savefig(full_save_as, dpi=200)
        print(f"{full_save_as} saved")
    plt.show()
    
    # full_title = f"Expected Turns till Death"
    # if title:
    #     full_title = f"{full_title}\n{title}"
    # if save_as:
    #     full_save_as = f"turns-{save_as}.jpg"


    return df

def plot2D_TAB(
        combat_gen,
        N=10000,
        num_swings=200,
        granularity=1,
        title="",
        save_as="",
        x=None,
        y=None,
        extra_func=None,
):

    df = TAB_table(
        combat_gen,
        N=N,
        num_swings=num_swings,
    )

    x2, y2 = df.columns.difference(["EKTD", "std_kills", "ETTD", "std_turns"])
    x = x or x2
    y = y or y2
    full_title = f"Expected Kills till Death"
    full_save_as = save_as
    if title:
        full_title = f"{full_title}\n{title}"
    if full_save_as:
        full_save_as = f"kills-{save_as}.jpg"

    make_heatmap(
        df[x],
        df[y],
        df["EKTD"],
        title=full_title,
        xlabel=x,
        ylabel=y,
        granularity=granularity,
        save_as=full_save_as,
        extra_func=extra_func,
    )

    full_title = f"Expected Turns till Death"
    if title:
        full_title = f"{full_title}\n{title}"
    if save_as:
        full_save_as = f"turns-{save_as}.jpg"

    make_heatmap(
        df[x],
        df[y],
        df["ETTD"],
        title=full_title,
        xlabel=x,
        ylabel=y,
        granularity=granularity,
        save_as=full_save_as,
        extra_func=extra_func,
    )
    return df

def TAB_table(
    combat_gen,
    N=10000,
    num_swings=200,
):
    data = list()
    for params, player, monster in combat_gen():
        success = False
        try_swings = num_swings
        while not success:
            try:
                EKTD, std_kills, ETTD, std_turns = TAB(player, monster, N, try_swings)
                row = params.copy()
                row.update(
                    dict(
                        EKTD=EKTD,
                        std_kills=std_kills,
                        ETTD=ETTD,
                        std_turns=std_turns,
                    )
                )
                data.append(row)
                success = True
            except (TooFewSwings, IndexError) as err:
                print(err)
                if try_swings >= 10 * num_swings:
                    raise TooFewSwings(
                        "Aborting to avoid memory error or possibly infinite loop."
                    )
                try_swings += num_swings
                print(f"Trying again with num_swings = {try_swings}")
        print(
            f"{params}: {EKTD:.1f} ± {std_kills:.1f} kills, {ETTD:.1f} ± {std_turns:.1f} turns"
        )
    df = pd.DataFrame(data)
    return df


def TAB(player, monster, N=10**4, num_swings=400):
    size = (N, num_swings)
    HP_monster = HP_instances(player, monster, size, check_for_death=False)
    HP_player = HP_instances(monster, player, size, check_for_death=True)

    # Find the indices where the player has died
    monster_swings = np.argmax(HP_player <= 0, axis=1)
    # estimated (10 aut) turns till death
    ettds = (monster_swings * monster.get("weapon_delay", 1.0)).astype(int)
    ETTD = ettds.mean()
    std_turns = ettds.std(ddof=1)

    # Find the amount of HP the monster has when the player died. This could be
    # negative. Divide by the monster's max HP to find an estimate of how many
    # monsters the player could have killed before dying himself.

    # HP_monster starts off at max HP, so (-HP_monster / monster["HP"] + 1) starts at 0
    player_swings = (
        monster_swings
        * monster.get("weapon_delay", 1.0)
        / player.get("weapon_delay", 1.0)
    ).astype(int)

    kills = ((-HP_monster / monster["HP"]) + 1)[range(N), player_swings]
    EKTD = kills.mean()
    std_kills = kills.std(ddof=1)
    return EKTD, std_kills, ETTD, std_turns


def HP_instances(attacker, defender, size, check_for_death=True):
    damage = np.zeros(size)
    try:
        for max_damage in attacker["damage"]:
            attack = attacker.copy()
            attack["damage"] = max_damage
            damage += damage_instances(attack, defender, size)
    except TypeError:
        damage += damage_instances(attacker, defender, size)



    mask = damage > 0
    # Reduce damage by Regen
    if "regen" not in defender and "HD" in defender:
        defender["regen"] = HD_to_regen(defender["HD"])

    damage -= mask * defender["regen"] * attacker.get("weapon_delay", 1.0)

    # Make sure damage is greater than or equal 0
    damage = np.maximum(damage, 0)

    # For each row (i.e. for each simulation), calculate the cumulative damage amounts
    total_damage = damage.cumsum(axis=1)

    # HP is an array representing the defender's hit points after taking damage
    HP = defender["HP"] - total_damage

    # Make sure that at some point in each simulation defender has died
    if check_for_death and not (HP <= 0).any(axis=1).all():
        raise TooFewSwings(
            f"""Defender ({defender["AC"]}/{defender["EV"]}/{defender["SH"]}) did not die after {size[1]} swings. Consider decreasing HP or increasing num_swings"""
        )

    return HP


def damage_instances(attacker, defender, size):
    if "HD" in attacker:
        # For monster
        to_hit = HD_to_hit(attacker["HD"], skilled=attacker.get("skilled", False))
        max_con_block = HD_to_con_block(attacker["HD"])
        BLOCK = possible_shield_outcomes(max_con_block, defender["SH"])
        block = np.random.choice(BLOCK.ravel(), size=size)
    else:
        # For player
        to_hit = calc_pre_roll_to_hit(attacker, size)
        max_con_block = to_hit_to_con_block(to_hit)
        con_block = CU.random2(max_con_block, size)
        SH = defender["SH"]
        pro_block1 = CU.random2(4 * SH, size)
        pro_block2 = CU.random2(4 * SH + 1, size)
        pro_block = (pro_block1 + pro_block2) // 6 - 1
        block = pro_block >= con_block

    hit = sample_hit_outcomes(to_hit, defender["EV"], size)
    max_damage = attacker["damage"] + attacker.get("slaying", 0)
    DAMAGE = possible_damage_outcomes(
        max_damage, defender["AC"], gdr=defender.get("gdr", False)
    )
    damage = np.random.choice(DAMAGE.ravel(), size=size)
    if attacker.get("brand") in ("flaming", "freezing"):
        f_mask = stats.bernoulli.rvs(0.25, size=size).astype(bool)
        damage[f_mask] = (damage[f_mask].astype(float) * 1.25).astype(int)
    elif attacker.get("brand") == "elec":
        elec_damage = np.random.randint(low=8, high=20 + 1, size=size)
        elec_mask = stats.bernoulli.rvs(0.25, size=size).astype(bool)
        damage[elec_mask] += elec_damage[elec_mask]
    elif attacker.get("brand") == "spectral":
        damage = (damage.astype(float)*1.7).astype(int)
        
    damage[block | ~hit] = 0
    return damage


def calc_pre_roll_to_hit(attacker, size):
    """
    See calc_pre_roll_to_hit from attack.cc
    """
    mhit = 15 + attacker["DEX"] // 2
    mhit += CU.random2_div(attacker["skills"]["Fighting"] * 100, 100, size)
    wpn_skill = item_attack_skill(attacker["weapon_type"])
    mhit += CU.random2_div(attacker["skills"][wpn_skill] * 100, 100, size)
    mhit += WEAPON_TO_HIT_BONUS[attacker["weapon_type"]]
    mhit += attacker["slaying"]
    mhit = mhit.clip(min=1)
    mhit = np.random.randint(mhit)
    return mhit

def to_hit_to_con_block(to_hit):
    return 15 + to_hit // 2


def sample_hit_outcomes(to_hit, EV, size):
    if not isinstance(to_hit, np.ndarray):
        # For monsters
        HIT = possible_hit_outcomes(to_hit, EV)
        result = np.random.choice(HIT.ravel(), size)
    else:
        # For players
        assert size == to_hit.shape
        # skipping post_roll_to_hit_modifiers -- which handles things like
        # confusion, invisibility, backlight and umbra
        roll1 = CU.random2(2 * EV, size) if EV else np.zeros(size)
        roll2 = CU.random2(2 * EV + 1, size)
        result = to_hit >= (roll1 + roll2) // 2

    mask = np.random.randint(100, size=size) < MIN_HIT_MISS_PERCENTAGE
    auto_hit_miss = np.random.randint(2, dtype=bool, size=mask.sum())
    result[mask] = auto_hit_miss
    return result


def possible_damage_outcomes(max_damage, AC, gdr=True):
    AC_roll = np.arange(AC + 1)
    damage_roll = np.arange(max_damage + 1)
    half_ac = [AC // 2, (AC + 1) // 2]
    AC_ROLL, DAMAGE_ROLL, HALF_AC = np.meshgrid(
        AC_roll, damage_roll, half_ac, indexing="ij"
    )
    if gdr:
        gdr = int(16 * AC**0.25) * max_damage // 100
        AC_ROLL = np.maximum(AC_ROLL, np.minimum(gdr, HALF_AC))
    DAMAGE_ROLL = np.maximum(DAMAGE_ROLL - AC_ROLL, 0)
    return DAMAGE_ROLL


def HD_to_regen(HD, class_regen_rate=1):
    """
    Returns average HP/turn regen rate
    """
    divider = max(((15 - HD) / 4), 1)
    rate = max((HD / divider), 1) / 25 * class_regen_rate
    rate = max(min(rate, 1), 0)
    return rate


def HD_to_hit(HD, skilled=False):
    # See fight.cc mon_to_hit_base
    # Set skilled=True if monster has the fighter *or* archer flag
    to_hit = int(18 + HD * 1.5 + (HD if skilled else 0))
    return to_hit


def HD_to_con_block(HD):
    """
    See fight.cc mon_shield_bypass
    """
    return 15 + HD * 2 // 3


def possible_shield_outcomes(max_con_block, SH):
    """
    Returns a boolean array of possible outcomes.
    True means the attack was blocked
    See attack.cc attack::attack_shield_blocked
    """
    con_block = np.arange(max_con_block)
    pro_block1 = np.arange(4 * SH) if SH else np.zeros(1)
    pro_block2 = np.arange(4 * SH + 1)
    CON_BLOCK, PRO_BLOCK1, PRO_BLOCK2 = np.meshgrid(
        con_block, pro_block1, pro_block2, indexing="ij"
    )
    PRO_BLOCK = (PRO_BLOCK1 + PRO_BLOCK2) // 6 - 1
    PROB = PRO_BLOCK >= CON_BLOCK
    return PROB


def possible_hit_outcomes(to_hit, EV):
    """
    Returns a boolean array of all possible outcomes.
    This DOES NOT INCLUDE the chance of automatic hits or misses
    True means the attack was a hit
    """
    to_land = np.arange(to_hit + 1)
    ev_roll1 = np.arange(2 * EV) if EV else np.zeros(1)
    ev_roll2 = np.arange(2 * EV + 1)
    TO_LAND, EV_ROLL1, EV_ROLL2 = np.meshgrid(
        to_land, ev_roll1, ev_roll2, indexing="ij"
    )
    HIT = TO_LAND >= (EV_ROLL1 + EV_ROLL2) // 2
    return HIT


def interpolate2d(x, y, z, granularity):

    # Interpolate the data to these grid points
    xi = np.linspace(x.min(), x.max(), int((x.max() - x.min()) * granularity) + 1)
    yi = np.linspace(y.min(), y.max(), int((y.max() - y.min()) * granularity) + 1)
    XI, YI = np.meshgrid(xi, yi, indexing="ij")
    ZI = interpolate.griddata((x, y), z, (XI, YI), rescale=True, method="cubic")
    return xi, yi, XI, YI, ZI


def make_heatmap(
    x,
    y,
    z,
    title="",
    xlabel="",
    ylabel="",
    granularity=1,
    xlim=None,
    ylim=None,
    levels=None,    
    save_as="",
    extra_func=None,
):
    # Increase granularity for finer interpolation
    xi, yi, XI, YI, ZI = interpolate2d(x, y, z, granularity)

    # make contour lines
    if levels is None:
        levels = np.arange(0, int(z.max().round()) + 1, 1)
        # levels = np.unique(np.linspace(0, int(z.max().round()) + 1, 20).astype(int))
    qcs = plt.contour(XI, YI, ZI, levels, linewidths=0.5, colors="black")
    plt.clabel(qcs, inline=True, fontsize=10)

    # make a heat map
    plt.pcolormesh(XI, YI, ZI, cmap=plt.get_cmap("rainbow"))
    plt.colorbar()

    # draw extra stuff if extra_func was supplied
    if extra_func is not None:
        extra_func(x, y, z, ZI, axes=[xlabel, ylabel], granularity=granularity)

    # draw straight dotted lines
    # w = int(xi.max() + yi.max() + 0.5)
    # for wi in range(1, w, 3):
    #     plt.plot((wi, 0), (0, wi), "k--")

    if xlim is None:
        xlim = (xi.min(), xi.max())
    if ylim is None:
        ylim = (yi.min(), yi.max())
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_as:
        plt.savefig(save_as, dpi=200)
        print(f"{save_as} saved")
    plt.show()


def make_hp_table(racial_modifier=0):
    fighting_skill = np.arange(0, 28).astype("int")
    xp = np.arange(0, 28).astype("int")
    F, XP = np.meshgrid(fighting_skill, xp, indexing="ij")
    BASE_HP = XP * 11 // 2 + 8
    BONUS = ((XP * F * 5 / 70) + (F * 3 + 1) / 2).astype(int)
    HP = BASE_HP + BONUS
    HP = (HP.astype(float) * (1 + racial_modifier)).astype(int)
    hp_table = pd.DataFrame(HP, index=fighting_skill, columns=xp)

    # TODO: handle MUT_ROBUST, MUT_RUGGED_BROWN_SCALES, MUT_FRAIL
    return hp_table


def find_optimal_skilling_path(
        combat_gen, axes, title="", save_as="", N=10**4, num_swings=400, granularity=2
):
    # make ETTD table
    df = TAB_table(combat_gen, N=N, num_swings=num_swings)

    # add XP column to df
    XP = np.array(
        [
            calc_XP({ax: coord / 10 for ax, coord in zip(axes, coords)})
            for coords in df[axes].values
        ]
    )
    # XP = np.log(XP + 1)
    df["XP"] = XP

    # interpolate ETTD and XP
    grid_points, XP = interpolate_df(df, axes, "XP", granularity=granularity)
    df2 = pd.DataFrame(grid_points, columns=axes)
    df2["XP"] = XP
    grid_points, df2["ETTD"] = interpolate_df(df, axes, "ETTD", granularity=granularity)
    
    # Use gradient to find optimal path
    grid_shape = [df2[ax].unique().size for ax in axes]
    ETTD = df2["ETTD"].values.reshape(*grid_shape)
    XP = df2["XP"].values.reshape(*grid_shape)
    ETTD_gradients = np.gradient(ETTD)
    XP_gradients = np.gradient(XP)

    location = [0] * len(axes)
    path_indices = [location.copy()]
    while any((coord < size - 1) for coord, size in zip(location, grid_shape)):
        # XP can be much larger than ETTD, so let's take the ratio of XP/ETTD
        # So small gradient is better than large gradient
        gradients = [
            abs(delta_XP[tuple(location)] / delta_ETTD[tuple(location)])
            for delta_XP, delta_ETTD in zip(XP_gradients, ETTD_gradients)
        ]
        # gradients = list()
        # for delta_XP, delta_ETTD in zip(XP_gradients, ETTD_gradients):
        #     print(f"abs({delta_XP[tuple(location)]} / {delta_ETTD[tuple(location)]}) = {abs(delta_XP[tuple(location)] / delta_ETTD[tuple(location)])}")
        #     gradients.append(abs(delta_XP[tuple(location)] / delta_ETTD[tuple(location)]))
        for idx in np.argsort(gradients):
            if location[idx] >= grid_shape[idx] - 1:
                continue
            else:
                # print(f"{gradients} -> {gradients[idx]}")
                location[idx] += 1
                path_indices.append(location.copy())
                break
    path_indices = np.array(path_indices)
    path_skills = df2.iloc[np.ravel_multi_index(path_indices.T, grid_shape)]

    # plot
    for col in axes:
        plt.plot(path_skills[col].values / 10, label=col)
    plt.ylabel("skill level")
    plt.xlabel("steps")
    ymax = int(path_skills[axes].values.max() / 10)
    plt.yticks(range(0, ymax + 1, 3))
    plt.legend()
    plt.title(title)
    plt.grid()
    if save_as:
        plt.savefig(save_as, dpi=200)
        print(f"{save_as} saved")
    plt.show()

    return path_skills

def calc_XP(skills):
    for sk, amount in skills.items():
        CS.set_skill_level(sk, amount, quiet=True, base_only=False)
    return CS.you.total_experience

def interpolate_df(df, axes, col, granularity=2):
    coarse_coords = tuple(np.unique(df[ax].values) for ax in axes)
    coarse_values = df[col].values.reshape(*(coord.size for coord in coarse_coords))
    fine_grid = np.meshgrid(
        *[
            np.linspace(
                coord.min(),
                coord.max(),
                (len(coord) - 1) * granularity + 1,
            ).round().astype(int)
            for coord in coarse_coords
        ],
        indexing="ij",
    )

    # make long list of grid points
    fine_grid_points = np.vstack([fg.ravel() for fg in fine_grid]).T
    fine_values = interpolate.interpn(
        coarse_coords, coarse_values, fine_grid_points, method="linear"
    )
    return fine_grid_points, fine_values


def base_ac_from(you, scale):
    base = you["total_parm_ac"] * scale
    AC = base * (440 + you["skills"]["Armour"] * 20) // 440
    AC *= you["AC_multiplier"]
    return AC

def base_ac(you, scale):
    AC = base_ac_from(you, 100)
    AC += you["AC_bonus"]
    return AC * scale // 100

def unadjusted_body_armour_penalty(you, archery=False):

    rfactor = 3 if you["SPARM_ARCHERY"] else 1
    return max(0, - PARM_EVASION[you["armour_type"]] // 10 // rfactor - you["MUT_STURDY_FRAME"] * 2)

def _player_aux_evasion_penalty(you, scale):
    piece_armour_evasion_penalty = 0
    for item in you["aux_armour"]:
        penalty = (- PARM_EVASION[item]) // 3
        if (penalty > 0):
            piece_armour_evasion_penalty += penalty;

    return piece_armour_evasion_penalty * scale // 10
        
    
def adjusted_body_armour_penalty(you, scale, archery=False):
    base_ev_penalty = unadjusted_body_armour_penalty(you, archery)
    return (2 * base_ev_penalty * base_ev_penalty * (450 - you["skills"]["Armour"] * 10) * scale 
            // (5 * (you["STR"] + 3)) // 450)


def adjusted_shield_penalty(you, scale):
    if (not you["shield_type"]):
        return 0

    base_shield_penalty = -PARM_EVASION[you["shield_type"]] // 10
    return (2 * base_shield_penalty * base_shield_penalty
            * (270 - you["skills"]["Shields"] * 10) * scale
            // (25 + 5 * you["STR"]) // 270)


def _player_evasion_size_factor(you):
    return dict(tiny=6, little=4, small=2, medium=0, large=-2, giant=-4)[you["body_size"]]

def _player_armour_adjusted_dodge_bonus(you, scale):
    dodge_bonus = ((800 + you["skills"]["Dodging"] * 10 * you["DEX"] * 8) * scale
        // (20 - _player_evasion_size_factor(you)) // 10 // 10)

    armour_dodge_penalty = unadjusted_body_armour_penalty(you) - 3
    if (armour_dodge_penalty <= 0):
        return dodge_bonus

    STR = max(1, you["STR"])
    if (armour_dodge_penalty >= STR):
        return dodge_bonus * STR // (armour_dodge_penalty * 2)
    return dodge_bonus - dodge_bonus * armour_dodge_penalty // (STR * 2)


def _player_evasion(you, final_scale):

    size_factor = _player_evasion_size_factor(you)
    scale = 100
    size_base_ev = (10 + size_factor) * scale

    # Calculate 'base' evasion from all permanent modifiers
    natural_evasion = (
        size_base_ev
        + _player_armour_adjusted_dodge_bonus(you, scale)
        - adjusted_body_armour_penalty(you, scale)
        - adjusted_shield_penalty(you, scale)
        - _player_aux_evasion_penalty(you, scale)
        + you["EV_bonus"] * scale)

    if (you["form"] == "statue"):
        natural_evasion = natural_evasion * 4 // 5

    return (natural_evasion * final_scale) // scale

def evasion(you, can_see_attacker=True):
    base_evasion = CU.div_rand_round(_player_evasion(you, 100), 100)
    invis_penalty = 0 if can_see_attacker else 10 
    return base_evasion - invis_penalty

def _sh_from_shield(you):
    base_shield = PARM_AC[you["shield_type"]] * 2
    # bonus applied only to base, see above for effect:
    shield = base_shield * 50
    shield += base_shield * you["skills"]["Shields"] * 5 // 2
    shield += you["skills"]["Shields"] * 38
    shield += 3 * 38
    shield += you["DEX"] * 38 * (base_shield + 13) // 26
    return shield

def player_shield_class(you, scale, random):
    """
    Calculate the SH value used internally.
    Exactly *twice* the value displayed to players, for legacy reasons.
    """
    shield = 0

    if (you["shield_type"] in ("buckler", "kite", "tower")):
        shield += _sh_from_shield(you)

    # multiply by 200 since the returned value is twice the displayed value
    shield += you["SH_bonus"] * 200

    return CU.div_rand_round(shield * scale, 100) if random else ((shield * scale) // 100)

def player_displayed_shield_class(you, scale):
    return player_shield_class(you, scale, False) // 2

def is_crossbow(weapon):
    return weapon in ("hand_cannon", "arbalest", "triple_crossbow")

def weapon_adjust_delay(you, base, random):
    brand = you["brand"]
    if (brand == "speed"):
        return CU.div_rand_round(base * 2, 3) if random else (base * 2) // 3
    if (brand == "heavy"):
        return CU.div_rand_round(base * 3, 2) if random else (base * 3) // 2
    return base;
    
def weapon_min_delay(you, check_speed):
    """How fast will this weapon get from your skill training?
    @param weapon the weapon to be considered.
    @param check_speed whether to take it into account if the weapon has the speed brand.
    @return How many aut the fastest possible attack with this weapon would take.
    """
    weapon = you["weapon_type"]
    base = PWPN_SPEED[weapon]

    if (weapon == "UNRAND_WOODCUTTERS_AXE"):
        return base

    min_delay = base//2

    # Short blades can get up to at least unarmed speed.
    if (item_attack_skill(weapon) == "Short Blades" and min_delay > 5):
        min_delay = 5

    # All weapons have min delay 7 or better
    if (min_delay > 7):
        min_delay = 7

    # ...except crossbows...
    if (is_crossbow(weapon) and min_delay < 10):
        min_delay = 10

    # ... and unless it would take more than skill 27 to get there.
    # Round up the reduction from skill, so that min delay is rounded down.
    min_delay = max(min_delay, base - (MAX_SKILL_LEVEL + 1) // 2)

    if (check_speed):
        min_delay = weapon_adjust_delay(you, min_delay, false)

    # never go faster than speed 3 (ie 3.33 attacks per round)
    if (min_delay < 3):
        min_delay = 3

    return min_delay
    
def weapon_min_delay_skill(you):
    weapon_type = you["weapon_type"]
    speed = PWPN_SPEED[weapon_type]
    mindelay = weapon_min_delay(you, False);
    return (speed - mindelay) * 2;


def item_attack_skill(weapon_type):
    return ("Axes" if weapon_type in ("war_axe", "battleaxe", "broad_axe", "hand_axe") else
            "Polearms" if weapon_type in ("spear", "trident", "demon_trident", "partisan", "halberd", "bardiche", "glaive", "trishula", "halberd") else
            "Short Blades" if weapon_type in ("dagger", "short_sword", "quick_blade", "rapier", "athame") else
            "Long Blades" if weapon_type in ("demon_blade", "falchion", "long_sword", "scimitar", "double_sword", "triple_sword", "great_sword", "eudemon_blade") else
            "Maces & Flails" if weapon_type in ("mace", "flail", "morningstar", "eveningstar", "demon_whip", "giant_spiked_club", "giant_club", "great_mace", "dire_flail", "sacred_scourge", "club") else
            "Ranged" if weapon_type in ("sling", "shortbow", "orcbow", "longbow", "arbalest", "triple_crossbow", "hand_cannon") else
            "Staves" if weapon_type in ("staff", "quarterstaff", "lajatang") else
            "Unknown weapon skill"
            )

    
def attack_delay_with(you):
    """
    Returns attack speed in auts
    """
    # FIX ME: UC, thrown and ranged weapon delays have not been implemented
    DELAY_SCALE = 20
    wpn_skill = item_attack_skill(you["weapon_type"])
    # Cap skill contribution to mindelay skill, so that rounding
    # doesn't make speed brand benefit from higher skill.
    wpn_sklev = min(you["skills"][wpn_skill] * 10, weapon_min_delay_skill(you) * 10)

    attk_delay = PWPN_SPEED[you["weapon_type"]]
    attk_delay -= wpn_sklev / DELAY_SCALE

    # unlike the C++ code, I want random=False so I get a "best" decimal value for attk_delay
    attk_delay = weapon_adjust_delay(you, attk_delay, random=False)

    # At the moment it never gets this low anyway.
    attk_delay = max(attk_delay, 3)

    attk_delay += adjusted_shield_penalty(you, DELAY_SCALE) / DELAY_SCALE

    # FIXME?: Finesse has not been implemented

    return max((attk_delay * you["player_speed"] / 10), 1)
    
def player_regen(you):
    return (20 + you["HP"] // 6) / 100 + you["bonus_regen"]

def get_real_mp(you):
    scale = 100
    spellcasting = you["skills"]["Spellcasting"] * scale
    scaled_xl = you["XL"] * scale

    # the first 4 experience levels give an extra .5 mp up to your spellcasting
    # the last 4 give no mp
    enp = min(23 * scale, scaled_xl)
    spell_extra = spellcasting
    invoc_extra = you["skills"]["Invocations"] * scale // 2
    highest_skill = max(spell_extra, invoc_extra)
    enp += highest_skill + min(8 * scale, min(highest_skill, scaled_xl)) // 2

    # FIXME: MUT_HIGH_MAGIC / MUT_LOW_MAGIC not implemented
    enp //= scale
    
    enp += CS._spec_stats.loc[you["race"], "MP"]

    # FIXME: mp_max_adj not implemented. What is "rotted" base?
    enp += you["bonus_MP"]

    if you["brand"] == "antimagic":
        enp /= 3
    enp = max(enp, 0)
    return enp

def player_mp_regen(you):
    # FIXME: MUT_HP_CASTING not implemented

    regen_amount = 7 + you["MP"] // 2

    # FIXME: MUT_MANA_REGENERATION not implemented -- affects Deep Elf

    # Amulets and artefacts bonuses are rolled into bonus_MP_regen

    # DUR_OOZE_REGEN not implemented
    # jelly_regen not implemented

    regen_amount += you["bonus_MP_regen"]

    return regen_amount / 100

def get_real_hp(you):

    hitp  = you["XL"] * 11 // 2 + 8
    hitp += (you["XL"] * you["skills"]["Fighting"] * 5 // 70
             + (you["skills"]["Fighting"] * 3 + 1) // 2)

    # Racial modifier.
    hitp *= 10 + CS.you.hp_modifier * 10
    hitp //= 10

    hitp += you["HP_bonus"]
    hitp *= you["HP_multiplier"]

    return max(1, hitp)

def brand_adjust_weapon_damage(base_dam, brand, random):
    if brand != "heavy":
        return base_dam
    return CU.div_rand_round(base_dam * 9, 5) if random else base_dam * 9 // 5

    
def weapon_uses_strength(wpn_skill):
    return wpn_skill not in ("Long Blades", "Short Blades", "Ranged")
    
def stat_modify_damage(you, damage, wpn_skill):
    # At 10 strength, damage is multiplied by 1.0
    # Each point of strength over 10 increases this by 0.025 (2.5%),
    # strength below 10 reduces the multiplied by the same amount.
    # Minimum multiplier is 0.01 (1%) (reached at -30 str).
    # Ranged weapons and short/long blades use dex instead.
    use_str = weapon_uses_strength(wpn_skill)
    attr = you["STR"] if use_str else you["DEX"]
    damage *= max(1.0, 75 + 2.5 * attr)
    damage //= 100

    return damage
    
def apply_weapon_skill(you, damage, wpn_skill, random):
    sklvl = you["skills"][wpn_skill] * 100
    damage *= 2500 + CU.maybe_random2(sklvl + 1, random)
    damage //= 2500
    return damage;

def apply_fighting_skill(you, damage, aux, random):
    base = 40 if aux else 30
    sklvl = you["skills"]["Fighting"] * 100

    damage *= base * 100 + CU.maybe_random2(sklvl + 1, random)
    damage //= base * 100

    return damage

def damage_rating(you):
    # FIXME: Throwing damage has not been implemented
    # FIXME: UC damage has not been implemented
    base_dam = PWPN_DAMAGE[you["weapon_type"]]

    # This is just SPWPN_HEAVY.
    post_brand_dam = brand_adjust_weapon_damage(base_dam, you["brand"], False)
    heavy_dam = post_brand_dam - base_dam
    extra_base_dam = heavy_dam
    skill = item_attack_skill(you["weapon_type"])

    stat_mult = stat_modify_damage(you, 100, skill)
    use_str = weapon_uses_strength(skill)
    # Throwing weapons and UC only get a damage mult from Fighting skill,
    # not from Throwing/UC skill.
    use_weapon_skill = (you["weapon_type"] != "UC")
    weapon_skill_mult = apply_weapon_skill(you, 100, skill, False) if use_weapon_skill else 100
    skill_mult = apply_fighting_skill(you, weapon_skill_mult, False, False)

    # slay bonuses and weapon plusses both improve to_hit and damage the same way
    # so I'm rolling both of them into you["slaying"]
    slaying = you["slaying"]
    plusses = slaying

    DAM_RATE_SCALE = 100
    rating = (base_dam + extra_base_dam) * DAM_RATE_SCALE
    rating = stat_modify_damage(you, rating, skill)
    if (use_weapon_skill):
        rating = apply_weapon_skill(you, rating, skill, False)
    rating = apply_fighting_skill(you, rating, False, False)
    rating //= DAM_RATE_SCALE
    rating += plusses
    return rating
    
params = dict()
params["gnoll"] = dict(
    damage=(6,),
    HD=2,
    skilled=False,
    AC=2,
    EV=9,
    SH=0,
    HP=13,
)

params["yak"] = dict(
    damage=(18,),
    HD=7,
    skilled=False,
    AC=4,
    EV=7,
    SH=0,
    HP=39,
)

params["bunyip"] = dict(
    damage=(40, 40, 40),
    HD=12,
    skilled=False,
    AC=6,
    EV=10,
    SH=0,
    HP=80,
    weapon_delay=3.0,
)

params["death yak"] = dict(
    damage=(30,),
    HD=14,
    skilled=False,
    AC=9,
    EV=5,
    SH=0,
    HP=77,
)

params["hydra"] = dict(
    damage=(18,),
    HD=13,
    skilled=False,
    AC=0,
    EV=5,
    SH=0,
    HP=72,
)

params["fire dragon"] = dict(
    damage=(20, 13, 13),
    HD=12,
    skilled=False,
    AC=10,
    EV=8,
    SH=0,
    HP=90,
)

params["titan"] = dict(
    damage=(55,),
    HD=20,
    skilled=True,
    AC=10,
    EV=3,
    SH=0,
    HP=110,
)

# 33% attack speed
params["juggernaut"] = dict(
    damage=(
        80,
        40,
    ),
    HD=20,
    skilled=True,
    AC=20,
    EV=5,
    SH=0,
    HP=170,
    attack_delay=3.0,
)

params["orb guardian"] = dict(
    damage=(45,), HD=15, skilled=False, AC=13, EV=13, SH=0, HP=83, attack_delay=1 / 1.4
)

