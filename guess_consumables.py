#!/usr/bin/env python
import itertools as IT
import string
import math
import pandas as pd


rarity = dict(
    potions=dict(very_common=192, common=105, uncommon=67, rare=35, very_rare=2),
    scrolls=dict(very_common=200, common=100, uncommon=36, rare=15, very_rare=9),
)

ITEM_RARITY = dict(
    potions=dict(
        curing="very_common",
        heal_wounds="common",
        enlightenment="uncommon",
        haste="uncommon",
        lignify="uncommon",
        attraction="uncommon",
        moonshine="uncommon",
        might="uncommon",
        brilliance="uncommon",
        mutation="uncommon",
        invisibility="rare",
        resistance="rare",
        magic="rare",
        berserk_rage="rare",
        cancellation="rare",
        ambrosia="rare",
        experience="very_rare",
    ),
    scrolls=dict(
        identify="very_common",
        teleportations="common",
        amnesia="uncommon",
        noise="uncommon",
        enchant_armour="uncommon",
        enchant_weapon="uncommon",
        revelation="uncommon",
        fear="uncommon",
        fog="uncommon",
        blinking="uncommon",
        immolation="uncommon",
        poison="uncommon",
        vulnerability="uncommon",
        butterflies="rare",
        summoning="rare",
        silence="rare",
        brand_weapon="rare",
        torment="rare",
        acquirement="very_rare",
    ),
)


def guess_consumables(quantity, known_potions, known_scrolls):
    item_rarity = ITEM_RARITY.copy()
    for pot in known_potions:
        del item_rarity["potions"][pot]
    for scroll in known_scrolls:
        del item_rarity["scrolls"][scroll]
    result = dict()
    for item_type in quantity.keys():
        qty = quantity[item_type]
        irarity = {
            item: rarity[item_type][kind]
            for item, kind in item_rarity[item_type].items()
        }

        names = list(string.ascii_letters[: len(qty)])
        universe = dict()
        for possibility in IT.permutations(irarity.keys(), len(qty)):
            # compute the likelihood of being in this universe
            universe[possibility] = math.prod(
                [irarity[pot] ** num for pot, num, name in zip(possibility, qty, names)]
            )

        udf = pd.DataFrame(
            [list(possibility) + [lval] for possibility, lval in universe.items()],
            columns=list(names) + ["likelihood"],
        )
        udf["prob"] = udf["likelihood"] / sum(udf["likelihood"])
        prob = pd.DataFrame()
        for name, qi in zip(names, qty):
            prob[f"{qi}{name}"] = udf.groupby(name)["prob"].sum()
        prob.index.name = None
        prob *= 100
        result[item_type] = prob
    return result


if __name__ == "__main__":
    quantity = dict(potions=[1, 1, 2, 2], scrolls=[2, 1, 1])
    known_potions = ["curing", "haste"]
    known_scrolls = ["identify", "fear"]
    probs = guess_consumables(quantity, known_potions, known_scrolls)
    print("Each columns represents an unknown item.", end="\n" * 2)
    for item_type, prob in probs.items():
        print(f"Probable identity of unknown {item_type}:")
        print(prob.to_markdown(), end="\n" * 2)
