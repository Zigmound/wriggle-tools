#!/usr/bin/env python
import sys
import argparse
import numpy as np
import pandas as pd
import crawl_tab as CT
import crawl_skills as CS

pd.options.display.width = sys.maxsize
pd.options.display.max_columns = None
pd.options.display.max_rows = sys.maxsize


def breakpoints(race):
    you = {
        "skills": {"Fighting": 0, "Spellcasting": 3.3, "Invocations": 0},
        "XL": 1,
        "HP_bonus": 0,
        "HP_multiplier": 1,
        "MP_bonus": 0,
        "brand": "none",
        "race": race,
    }
    skill_levels = np.arange(0, 27.1, 0.1)
    XL = range(1, 28)

    CS.you = CS.Player(race)
    result = list()
    for skl, stat, calc in [
        ("Fighting", "HP", CT.get_real_hp),
        ("Spellcasting", "MP", CT.get_real_mp),
    ]:
        data = list()
        for xli in XL:
            you["XL"] = xli
            previous = -1
            for lvl in skill_levels:
                you["skills"][skl] = lvl
                statval = calc(you)
                if statval > previous:
                    data.append((xli, lvl, statval))
                    previous = statval
        df = pd.DataFrame(data, columns=["XL", skl, stat])
        df = df.pivot_table(index=skl, columns="XL", values=stat)
        result.append((race, stat, df))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--race", type=str, default="Human", help="race of the character"
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="print just the first 30 rows and 10 columns of each table",
    )
    parser.add_argument("--all", action="store_true", help="dump tables for every race")
    args = parser.parse_args()

    races = CS.SPECIES if args.all else [args.race]
    for race in races:
        result = breakpoints(race)
        for race, stat, df in result:
            print(f"{race} {stat} breakpoints (columns represent XL)")
            if args.short:
                df = df.iloc[:, :10]
                df = df.dropna(axis="rows", how="all")
                df = df.iloc[:30, :]
            df = df.fillna(-1).astype(int).astype(str)
            for col in df:
                df[col] = df[col].str.replace("-1", "")
            print(df.to_markdown(), end="\n" * 2)
