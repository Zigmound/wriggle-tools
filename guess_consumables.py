#!/usr/bin/env python
import itertools as IT
import math
import string
import numpy as np
import pandas as pd

try:
    from thewalrus import perm

    def guess_potions(qty, known_items):
        return guess_consumables(qty, known_items, potions, potion_weights)

    def guess_scrolls(qty, known_items):
        return guess_consumables(qty, known_items, scrolls, scroll_weights)

except ModuleNotFoundError:

    def perm(M):
        """
        Author: lesshaste
        Date: 2017-03-09
        https://github.com/scipy/scipy/issues/7151
        """
        n = M.shape[0]
        d = np.ones(n)
        j = 0
        s = 1
        f = np.arange(n)
        v = M.sum(axis=0)
        p = np.prod(v)
        while j < n - 1:
            v -= 2 * d[j] * M[j]
            d[j] = -d[j]
            s = -s
            prod = np.prod(v)
            p += s * prod
            f[0] = 0
            f[j] = f[j + 1]
            f[j + 1] = j + 1
            j = f[0]
        return p / 2 ** (n - 1)

    def guess_potions(qty, known_items):
        if len(qty) < 6:
            return guess_consumables_bruteforce(
                qty, known_items, potions, potion_weights
            )
        else:
            return guess_consumables(qty, known_items, potions, potion_weights)

    def guess_scrolls(qty, known_items):
        if len(qty) < 6:
            return guess_consumables_bruteforce(
                qty, known_items, scrolls, scroll_weights
            )
        else:
            return guess_consumables(qty, known_items, scrolls, scroll_weights)


def guess_consumables_bruteforce(qty, known_items, items, item_weights):
    unknown_knowns = set(known_items).difference(items)
    if any(unknown_knowns):
        raise ValueError(f"unknown items: {unknown_knowns} (check spelling)")
    items, item_weights = zip(
        *[
            (item, weight)
            for item, weight in zip(items, item_weights)
            if not item in known_items
        ]
    )
    irarity = dict(zip(items, item_weights))

    universe = dict()
    for possibility in IT.permutations(items, len(qty)):
        # compute the likelihood of being in this universe
        universe[possibility] = math.prod(
            [irarity[item] ** num for item, num in zip(possibility, qty)]
        )
    names = list(string.ascii_letters[: len(qty)])
    udf = pd.DataFrame(
        [list(possibility) + [lval] for possibility, lval in universe.items()],
        columns=list(names) + ["likelihood"],
    )
    udf["prob"] = udf["likelihood"] / sum(udf["likelihood"])
    prob = pd.DataFrame()
    seen = set()
    for name, qi in zip(names, qty):
        if qi not in seen:
            seen.add(qi)
            prob[qi] = udf.groupby(name)["prob"].sum()

    prob.index.name = None
    prob *= 100
    prob = prob.reindex(items)
    return prob


def guess_consumables(qty, known_items, items, item_weights):
    qty = np.asarray(qty)
    items = np.asarray(items)
    item_weights = np.asarray(item_weights)
    known_items = np.asarray(known_items)
    unknown_knowns = ~np.isin(known_items, items)
    if unknown_knowns.any():
        raise ValueError(
            f"unknown items: {known_items[unknown_knowns]} (check spelling)"
        )
    mask = np.isin(items, known_items)
    items = items[~mask]
    item_weights = item_weights[~mask]
    p = item_weights.astype("float")
    p /= p.sum()

    n = p.size
    v = np.zeros(n)
    k = qty.size
    v[:k] = qty

    M = p[:, None] ** v[None, :]
    permM = perm(M)

    uniq_qty, index = np.unique(qty, return_index=True)
    prob = np.zeros((n, index.size))
    for i in range(n):
        for j_pos, j in enumerate(index):
            row_mask = np.arange(n) != i
            col_mask = np.arange(n) != j
            Mij = M[np.ix_(row_mask, col_mask)]
            prob[i, j_pos] = p[i] ** v[j] * perm(Mij) / permM
    prob = prob * 100
    df = pd.DataFrame(prob, index=items, columns=uniq_qty)

    return df


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
        lignification="uncommon",
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
        teleportation="common",
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


potions, potion_weights = zip(
    *[
        (item, rarity["potions"][rarity_type])
        for item, rarity_type in ITEM_RARITY["potions"].items()
    ]
)
scrolls, scroll_weights = zip(
    *[
        (item, rarity["scrolls"][rarity_type])
        for item, rarity_type in ITEM_RARITY["scrolls"].items()
    ]
)


if __name__ == "__main__":
    quantity = [1, 1, 2, 3, 1]
    known_potions = []
    df = guess_potions(quantity, known_potions)
    print("Each columns represents an unknown item.", end="\n" * 2)
    print(df.to_markdown(), end="\n" * 2)

    quantity = [1, 1, 1, 1, 5]
    known_scrolls = ["identify", "fear"]
    df = guess_scrolls(quantity, known_scrolls)
    print("Each columns represents an unknown item.", end="\n" * 2)
    print(df.to_markdown(), end="\n" * 2)
