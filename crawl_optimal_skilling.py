#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
import crawl_tab as CT
import crawl_skills as CS

pd.options.display.width = sys.maxsize
pd.options.display.max_columns = None

def bang_for_xp_buck(race, monster, target, configs):
    # find the XP required to reach the target skill level
    CS.you = CS.Player(race)

    df = pd.DataFrame(target[1:], columns=target[0])
    df["skl"] = [CS.long_to_short[val] for val in df["skill"].values]
    skills = dict(zip(df["skl"], df["current"].values))
    threshold_XP = 0
    for idx, row in df.iterrows():
        if row.target:
            skills2 = skills.copy()
            skills2[row.skl] = row.target
            min_XP = CT.calc_XP(skills2)
            skills2[row.skl] = row.target + 0.1
            max_XP = CT.calc_XP(skills2)
            # Find an amount of XP which is comfortably within
            # the range needed to achieve the target skill level
            threshold_XP += 0.8 * min_XP + 0.2 * max_XP

    # find the skill levels achievable for the same amount of XP
    result = list()
    for skill, level in skills.items():
        if skill not in (
            "Fgt",
            "Arm",
            "Ddg",
            "Shd",
            "MF",
            "Axs",
            "Pla",
            "Stv",
            "UC",
            "SBl",
            "LBl",
            "Spc",
            "Inv",
        ):
            # My simulator is not affected by the other skills, so we are blind to the effect
            # training these skills have on true EKTD/ETTD
            continue
        skills2 = skills.copy()
        while skills2[skill] <= 27:
            if CT.calc_XP(skills2) < threshold_XP:
                skills2[skill] += 0.1
            else:
                skills2[skill] -= 0.1
                break
        skills2[skill] += df.loc[df["skl"] == skill, "skill_bonus"]
        skills2[skill] = min(skills2[skill].item(), 27)
        result.append((skill, skills[skill], skills2[skill]))

    bfxb = pd.DataFrame(result, columns=["skill", "current", "target"])
    bfxb["diff"] = bfxb["target"] - bfxb["current"]
    return bfxb, skills

def find_best_skill(race, monster, target, configs, N=10**4, num_swings=400):
    bfxb, skills = bang_for_xp_buck(race, monster, target, configs)
    # Compute EKTD and ETTD for each config and for each skill
    result = list()
    for row in configs[1:]:
        for idx, skill_row in bfxb.iterrows():
            player = dict(
                gdr=True,
            )
            player["race"] = race
            config = dict(zip(configs[0], row))
            config["aux_armour"] = config["aux_armour"].split()
            player.update(config)

            skills2 = skills.copy()
            skills2[skill_row.skill] = skill_row.target
            player["skills"] = {
                CS.short_to_long[key]: val for key, val in skills2.items()
            }

            player["HP"] = CT.get_real_hp(player)

            player["total_parm_ac"] = CT.PARM_AC.get(player["armour_type"], 0)
            for item in player["aux_armour"]:
                player["total_parm_ac"] += CT.PARM_AC[item]
            player["AC"] = CT.base_ac(player, 100)
            player["EV"] = CT.evasion(player, 1)
            player["SH"] = CT.player_displayed_shield_class(player, 1)
            player["damage"] = CT.damage_rating(player)
            player["weapon_delay"] = CT.attack_delay_with(player) / 10
            player["regen"] = CT.player_regen(player)
            player["MP"] = CT.get_real_mp(player)
            player["MP_regen"] = CT.player_mp_regen(player)

            if config["spirit"]:
                # total hack
                player["HP"] += player["MP"]
                player["regen"] += player["MP_regen"]

            kills, std_kills, turns, std_turns = CT.TAB(
                player, monster, N=N, num_swings=num_swings
            )
            observation = dict(
                skill=skill_row.skill,
                skill_level=skill_row.target,
                kills=kills,
                std_kills=std_kills,
                turns=turns,
                std_turns=std_turns,
            )
            observation.update(player)
            result.append(observation)

    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values(by="kills", ascending=False)
    skill_df = pd.DataFrame(result_df["skills"].tolist(), index=result_df.index)
    del result_df["skills"]
    result_df = pd.merge(result_df, skill_df, left_index=True, right_index=True)
    columns = [
        "name",
        "skill",
        "skill_level",
        "kills",
        "std_kills",
        "turns",
        "std_turns",
        "HP",
        "MP",
        "AC",
        "EV",
        "SH",
        "damage",
        "weapon_delay",
        "regen",
        "MP_regen",
        "STR",
        "DEX",
        "XL",
        "HP_bonus",
        "HP_multiplier",
        "regen_bonus",
        "MP_bonus",
        "MP_regen_bonus",
        "AC_bonus",
        "AC_multiplier",
        "EV_bonus",
        "SH_bonus",
    ]
    anything_else = result_df.columns.difference(columns)
    columns = columns + anything_else.tolist()
    result_df = result_df[columns]
    return bfxb, result_df


if __name__ == "__main__":
    # configurable inputs
    race = "Octopode"
    monster = CT.params["titan"]

    # The third value on each row set the target skill levels.
    # Below, we'll compute the amount of XP required to reach the target skill levels.
    # If no value is set, the program will find the skill level attainable for the same amount of XP.

    target = [
        ["skill", "current", "target", "skill_bonus"],
        ["Fighting", 13, 18, 0],
        ["Polearms", 14, "", 0],
        ["Armour", 0, "", 0],
        ["Dodging", 18, "", 0],
        ["Shields", 19, "", 0],
        ["Spellcasting", 2, "", 0],
        ["Conjurations", 3, "", 0],
        ["Necromancy", 2, "", 0],
        ["Translocations", 10, "", 0],
        ["Invocations", 10, "", 0],
        ["Evocations", 8, "", 0],
    ]

    configs = [
        ("name", "dex", "evasion"),
        ("XL", 22, 22),
        ("STR", 45, 39),
        ("DEX", 27, 27),
        ("slaying", 8, 8),
        ("brand", "flaming", "flaming"),
        ("weapon_type", "demon_trident", "demon_trident"),
        ("armour_type", "none", "none"),
        ("aux_armour", "", ""),
        ("shield_type", "tower", "tower"),
        ("AC_multiplier", 1, 1),
        ("AC_bonus", 0, 0),
        ("EV_bonus", 5, 10),
        ("SH_bonus", 7, 7),
        ("HP_bonus", 0, 0),
        ("HP_multiplier", 1, 1),
        ("regen_bonus", 0, 0),
        ("MP_regen_bonus", 0, 0),
        ("MP_bonus", 0, 0),
        ("body_size", "medium", "medium"),
        ("SPARM_ARCHERY", 0, 0),
        ("MUT_STURDY_FRAME", 0, 0),
        ("spirit", 0, 0),
        ("player_speed", 10, 10),
        ("form", "none", "none"),
    ]

    configs = list(zip(*configs))

    skill_df, result_df = find_best_skill(race, monster, target, configs)
    print(skill_df, end="\n" * 2)
    print(result_df)
