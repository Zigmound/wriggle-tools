#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
import crawl_tab as CT
import crawl_skills as CS
pd.options.display.width = sys.maxsize
pd.options.display.max_columns = None

def find_best_skill(race, monster, bang, configs):
    # find the XP required to reach the target skill level
    CS.you = CS.Player(race)
    bfxb = pd.DataFrame(bang[1:], columns=bang[0])
    bfxb["skl"] = [CS.long_to_short[val] for val in bfxb["skill"].values]
    skills = dict(zip(bfxb["skl"], bfxb["current"].values))
    threshold_XP = 0
    for idx, row in bfxb.iterrows():
        if row.target:
            skills2 = skills.copy()
            skills2[row.skl] = row.target
            min_XP = CT.calc_XP(skills2)
            skills2[row.skl] = row.target + 0.1
            max_XP = CT.calc_XP(skills2)
            # Find an amount of XP which is comfortably within
            # the range needed to achieve the target skill level
            threshold_XP += 0.8 * min_XP + 0.2 * max_XP
    print(f"threshold_XP: {threshold_XP:.0f}")

    # find the skill levels achievable for the same amount of XP
    result = list()
    for skill, level in skills.items():
        if skill not in ("Fgt", "Arm", "Ddg", "Shd", "MF", "Axs", "Pla", "Stv", "UC",
                         "SBl", "LBl", "Spc", "Inv"):
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
        skills2[skill] += bfxb.loc[bfxb["skl"] == skill, "skill_bonus"]
        skills2[skill] = min(skills2[skill].item(), 27)
        result.append((skill, skills[skill], skills2[skill]))
    df = pd.DataFrame(result, columns=["skill", "current", "target"])
    df["diff"] = df["target"] - df["current"]

    print(df, end="\n" * 2)

    # Compute EKTD and ETTD for each config and for each skill 
    params = dict(N=10**4, num_swings=400)
    result = list()
    for row in configs[1:]:
        for idx, skill_row in df.iterrows():
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
            player["AC"] = CT.base_ac(player, 1)

            player["EV"] = CT.evasion(player, 1)
            player["SH"] = CT.player_displayed_shield_class(player, 1)
            player["weapon_delay"] = CT.attack_delay_with(player) / 10
            player["regen"] = (20 + player["HP"] / 6) / 100 + player["bonus_regen"]
            player["MP"] = CT.get_real_mp(player)
            player["MP_regen"] = 7 + player["MP"] / 2 + player["bonus_MP_regen"]

            if config["spirit"]:
                # total hack
                player["HP"] += player["MP"]
                player["regen"] += player["MP_regen"]
                
            # print(player)
            
            kills, std_kills, turns, std_turns = CT.TAB(player, monster, **params)
            observation = dict(config=config["name"], skill=skill_row.skill,
                               skill_level=skill_row.target, kills=kills, std_kills=std_kills,
                               turns=turns, std_turns=std_turns)
            observation.update(player)
            result.append(observation)

    result_df = pd.DataFrame(result)
    result_df["score"] = 10 * result_df["kills"] + result_df["turns"]
    result_df = result_df.sort_values(by="score", ascending=False)
    print()
    print(result_df)
    
if __name__ == "__main__":
    # configurable inputs
    race = "Octopode"
    monster = CT.params["fire dragon"]

    # The third values on each row set the target skill levels.
    # Below, we'll compute the amount of XP required to reach the target skill levels.
    # If no value is set, the program will find the skill level attainable for the same amount of XP.
    bang = [
        ("skill", "current", "target", "skill_bonus"),
        ("Fighting", 13, 18, 0),
        ("Polearms", 14, "", 0),
        ("Dodging", 9, "", 0),
        ("Armour", 0, "", 0),
        ("Shields", 10, "", 2),
        ("Spellcasting", 2, "", 0),
        ("Conjurations", 3, "", 0),
        ("Necromancy", 2, "", 0),
        ("Translocations", 10, "", 0),
        ("Invocations", 9, "", 0),
        ("Evocations", 5, "", 0),
    ]
    configs = [
        ("name", "char"),
        ("XL", 14),
        ("STR", 19),
        ("DEX", 23),
        ("damage", 18),
        ("slaying", 0),
        ("brand", "none"),
        ("weapon_type", "trident"),
        ("armour_type", "none"),
        ("aux_armour", ""),
        ("shield_type", "kite"),
        ("AC_multiplier", 1),
        ("AC_bonus", 0),
        ("EV_bonus", 0),
        ("SH_bonus", 0),
        ("bonus_regen", 0),
        ("bonus_MP_regen", 0),
        ("bonus_MP", 0),
        ("body_size", "medium"),
        ("SPARM_ARCHERY", 0),
        ("MUT_STURDY_FRAME", 0),
        ("spirit", 0),
        ("player_speed", 10),
        ("form", "none"),
    ]
    configs = list(zip(*configs))

    find_best_skill(race, monster, bang, configs)
