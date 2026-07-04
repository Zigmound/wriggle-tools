#!/usr/bin/env python
import math
from functools import lru_cache
import numpy as np
import pandas as pd

DEBUG = False
MAX_SKILL_LEVEL = 27
MAX_SKILL_COST_LEVEL = 27
APT_DOUBLE = 4
skill_name_pairs = [
    ("Arm", "Armour"),
    ("Ddg", "Dodging"),
    ("Sth", "Stealth"),
    ("Shd", "Shields"),
    ("Shp", "Shapeshifting"),
    ("Inv", "Invocations"),
    ("Evo", "Evocations"),
    ("Thr", "Throwing"),
    ("Fgt", "Fighting"),
    ("MF", "Maces & Flails"),
    ("Axs", "Axes"),
    ("Pla", "Polearms"),
    ("Stv", "Staves"),
    ("UC", "Unarmed Combat"),
    ("SBl", "Short Blades"),
    ("LBl", "Long Blades"),
    ("Rng", "Ranged"),
    ("Spc", "Spellcasting"),
    ("Coj", "Conjurations"),
    ("Hex", "Hexes"),
    ("Sum", "Summoning"),
    ("Nec", "Necromancy"),
    ("Fgr", "Forgecraft"),
    ("Trl", "Translocations"),
    ("Fir", "Fire"),
    ("Ice", "Ice"),
    ("Air", "Air"),
    ("Ear", "Earth"),
    ("Alc", "Alchemy"),
]
short_to_long = {pair[0]: pair[1] for pair in skill_name_pairs}
long_to_short = {pair[1]: pair[0] for pair in skill_name_pairs}

species_skill_aptitude = """
| species          | Arm | Ddg | Sth | Shd | Shp | Inv | Evo | Thr | Fgt | MF  | Axs | Pla | Stv | UC | SBl | LBl | Rng | Spc | Coj | Hex | Sum | Nec | Fgr | Trl | Fir | Ice | Air | Ear | Alc |
|------------------+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----|
| Armataur         |   2 |  -3 |   2 |   1 |  -2 |   0 |   0 |  -1 |  -1 |  -1 |  -2 |  -2 |  -1 | -1 |  -1 |  -1 |  -3 |  -2 |  -1 |  -1 |  -2 |  -2 |  -2 |   0 |  -1 |  -1 |  -1 |  -1 |  -1 |
| Gale Centaur     |   0 |  +1 |  +2 |  +1 |  -2 |  +2 |  -1 |  -2 |   0 |  -1 |  -2 |  -1 |  -1 | -1 |  -1 |  -1 |   0 |  -2 |  -1 |  -1 |  -2 |  -2 |  -2 |  +1 |  -1 |  -1 |  +2 |  -1 |  -2 |
| Barachi          |   2 |   1 |   0 |   1 |   0 |  -1 |   1 |   0 |   2 |   1 |   1 |   0 |   1 |  1 |   1 |   2 |   0 |   0 |   1 |   1 |   2 |  -1 |   1 |   1 |   1 |   2 |   1 |   0 |   1 |
| Coglin           |  -1 |  -1 |  -1 |  -3 |  -2 |  -2 |   3 |  -1 |   0 |  -1 |   0 |  -1 |  -1 | -1 |  -1 |   0 |  -1 |  -2 |  -1 |  -1 |   0 |   0 |   2 |   0 |  -1 |  -1 |  -1 |  -1 |   1 |
| Deep Elf         |  -2 |   2 |   3 |  -2 |   0 |   1 |   1 |   0 |  -2 |  -3 |  -2 |  -3 |   0 | -2 |   0 |  -1 |   3 |   3 |   1 |   3 |   1 |   2 |   1 |   1 |   1 |   1 |   1 |   1 |   1 |
| Demigod          |  -1 |  -1 |   0 |  -1 |  -2 |   0 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 | -1 |  -1 |  -1 |  -1 |  -2 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |
| Demonspawn       |  -1 |  -1 |   0 |  -1 |  -2 |   3 |   0 |  -1 |   0 |  -1 |  -1 |  -1 |  -1 | -1 |  -1 |  -1 |  -1 |  -1 |   0 |   0 |   0 |   1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |   0 |
| Djinni           |   0 |   1 |  -1 |   0 |  -2 |   0 |   0 |  -2 |   0 |  -2 |  -2 |  -2 |  -1 |  0 |  -1 |  -1 |  -2 |  11 |  11 |  11 |  11 |  11 |  11 |  11 |  11 |  11 |  11 |  11 |  11 |
| Draconian        |   0 |  -1 |   0 |   0 |  -1 |   1 |   0 |  -1 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |  -1 |   0 |  -1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| Draconian Black  |   0 |  -1 |   0 |   0 |  -1 |   1 |   0 |  -1 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |  -1 |   0 |  -1 |   0 |   0 |   0 |   0 |   0 |   0 |   2 |  -2 |   0 |
| Draconian Green  |   0 |  -1 |   0 |   0 |  -1 |   1 |   0 |  -1 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |  -1 |   0 |  -1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   2 |
| Draconian Grey   |   0 |  -1 |   0 |   0 |  -1 |   1 |   0 |  -1 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |  -1 |   0 |  -1 |   0 |   0 |   0 |   0 |   0 |   0 |  -2 |   2 |   0 |
| Draconian Pale   |   0 |  -1 |   0 |   0 |  -1 |   1 |   1 |  -1 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |  -1 |   0 |  -1 |   0 |   0 |   0 |   0 |   1 |   0 |   1 |   0 |   0 |
| Draconian Purple |   0 |  -1 |   0 |   0 |  -1 |   1 |   1 |  -1 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |   1 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| Draconian Red    |   0 |  -1 |   0 |   0 |  -1 |   1 |   0 |  -1 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |  -1 |   0 |  -1 |   0 |   0 |   0 |   0 |   2 |  -2 |   0 |   0 |   0 |
| Draconian White  |   0 |  -1 |   0 |   0 |  -1 |   1 |   0 |  -1 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |  -1 |   0 |  -1 |   0 |   0 |   0 |   0 |  -2 |   2 |   0 |   0 |   0 |
| Draconian Yellow |   0 |  -1 |   0 |   0 |  -1 |   1 |   0 |  -1 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |  -1 |   0 |  -1 |   0 |   0 |   2 |   0 |   0 |   0 |   0 |   0 |   0 |
| Felid            |   0 |   3 |   4 |   0 |  -2 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |   0 |  -1 |  -1 |   4 |   0 |   0 |  -1 |   4 |  -1 |  -1 |  -1 |  -1 |  -1 |
| Formicid         |   1 |  -1 |   3 |   3 |   0 |   2 |   1 |   0 |   1 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |   0 |   0 |  -1 |   2 |   0 |   0 |   0 |   2 |   0 |   0 |  -2 |   2 |   3 |
| Gargoyle         |   1 |  -2 |   2 |   1 |  -3 |   1 |  -1 |  -1 |   1 |   0 |  -1 |  -1 |   0 |  0 |  -1 |  -1 |   0 |  -1 |   1 |  -1 |  -1 |  -2 |  -1 |  -1 |   0 |   0 |  -2 |   2 |  -2 |
| Ghoul            |  -1 |  -1 |   2 |  -1 |   0 |   1 |  -1 |  -1 |   1 |  -1 |  -1 |  -1 |  -1 |  1 |  -1 |  -1 |  -1 |  -2 |  -2 |  -2 |  -1 |   0 |  -2 |  -1 |  -2 |   1 |  -2 |   1 |  -1 |
| Gnoll            |   8 |   8 |   8 |   8 |   7 |   9 |   8 |   8 |   8 |   8 |   8 |   8 |   8 |  8 |   8 |   8 |   8 |   8 |   6 |   6 |   6 |   6 |   6 |   6 |   6 |   6 |   6 |   6 |   6 |
| Human            |   0 |   0 |   1 |   0 |  -1 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |   0 |  -1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| Kobold           |  -2 |   2 |   4 |  -2 |  -1 |   1 |   2 |   1 |   1 |  -1 |  -2 |  -2 |  -1 |  0 |   3 |  -2 |   3 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
| Merfolk          |  -3 |   3 |   2 |   0 |   2 |   1 |   0 |   0 |   1 |  -2 |  -2 |   4 |  -2 |  1 |   2 |   2 |  -2 |  -1 |  -2 |   0 |   0 |  -2 |   0 |  -2 |  -3 |   1 |  -2 |  -2 |   3 |
| Minotaur         |   2 |   1 |  -1 |   2 |  -3 |   0 |  -1 |   0 |   2 |   2 |   2 |   2 |   2 |  1 |   1 |   2 |   1 |  -4 |  -3 |  -4 |  -3 |  -3 |  -2 |  -3 |  -3 |  -3 |  -3 |  -2 |  -3 |
| Mountain Dwarf   |   1 |  -3 |  -2 |   1 |  -2 |   3 |   1 |  -2 |   1 |   2 |   2 |   0 |   1 |  0 |  -2 |  -1 |  -2 |  -2 |  -1 |   0 |   0 |   1 |   2 |  -2 |   2 |  -1 |  -3 |   1 |  -2 |
| Mummy            |  -2 |  -2 |  -1 |  -2 |   0 |  -1 |  -2 |  -2 |   0 |  -2 |  -2 |  -2 |  -2 | -2 |  -2 |  -2 |  -2 |   2 |  -2 |  -1 |  -2 |   0 |  -2 |  -2 |  -2 |  -2 |  -2 |  -2 |  -2 |
| Naga             |  -2 |  -2 |   5 |  -1 |  -1 |   1 |   0 |  -1 |   0 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |  -1 |  -1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   2 |
| Octopode         |   0 |   0 |   4 |   0 |  -1 |   1 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |  0 |   0 |   0 |   0 |  -1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   1 |
| Oni              |  -1 |  -1 |  -2 |  -1 |  -1 |   2 |  -2 |   0 |   3 |   0 |   0 |   0 |   0 | -1 |  -1 |  -1 |  -3 |   1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |
| Poltergeist      |   0 |   1 |   5 |  -1 |   0 |  -1 |  -1 |   2 |  -1 |  -2 |  -1 |  -1 |  -2 | -3 |   1 |   0 |  -2 |  -1 |  -3 |   4 |   0 |   1 |  -1 |   0 |  -1 |   1 |   1 |  -1 |   1 |
| Revenant         |  -1 |  -1 |   2 |  -1 |   0 |   1 |  -1 |  -1 |   1 |  -1 |  -1 |  -1 |  -1 |  1 |  -1 |  -1 |  -3 |  -1 |  -1 |  -2 |  -1 |   0 |  -2 |  -1 |  -2 |   1 |  -2 |   1 |  -1 |
| Spriggan         |  -3 |   3 |   5 |  -3 |   2 |   0 |   3 |   0 |  -2 |  -3 |  -2 |  -3 |  -3 | -2 |   1 |  -2 |   0 |   2 |  -3 |   2 |  -2 |  -1 |  -2 |   4 |  -2 |  -2 |  -1 |  -1 |   1 |
| Tengu            |   1 |   1 |   1 |   0 |  -2 |  -1 |   0 |   0 |   0 |   1 |   1 |   1 |   1 |  1 |   1 |   1 |   1 |  -1 |   3 |  -3 |   2 |   1 |  -2 |  -2 |   1 |  -1 |   3 |  -3 |  -1 |
| Troll            |  -2 |  -2 |  -5 |  -1 |  -1 |  -1 |  -3 |  -1 |  -2 |  -1 |  -2 |  -2 |  -2 |  0 |  -2 |  -2 |  -4 |  -5 |  -3 |  -4 |  -3 |  -2 |  -3 |  -3 |  -3 |  -3 |  -4 |  -1 |  -3 |
| Vampire          |  -2 |   1 |   5 |  -1 |   0 |  -1 |  -1 |  -2 |  -1 |  -2 |  -1 |  -1 |  -2 |  1 |   1 |   0 |  -2 |  -1 |  -3 |   4 |   0 |   1 |   0 |  -2 |  -2 |   0 |   0 |   0 |   1 |
| Vine Stalker     |  -2 |  -2 |   3 |  -1 |  -1 |   0 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  -1 |  0 |  -1 |  -1 |  -1 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |   0 |
"""

species_stats = """
| species          | HP% | MP | Exp | WL |
|------------------+-----+----+-----+----|
| Gnoll            |   0 |  0 |   0 |  3 |
| Gargoyle         | -20 |  0 |   0 |  3 |
| Spriggan         | -30 |  1 |  -1 |  7 |
| Mountain Dwarf   |  10 |  0 |  -1 |  4 |
| Demonspawn       |   0 |  0 |  -1 |  3 |
| Octopode         | -10 |  0 |   0 |  3 |
| Poltergeist      | -10 |  0 |   0 |  4 |
| Revenant         |  10 |  1 |  -1 |  3 |
| Tengu            | -20 |  1 |   0 |  3 |
| Felid            | -30 |  1 |  -1 |  6 |
| Vine Stalker     | -30 |  1 |   0 |  5 |
| Kobold           | -20 |  0 |   1 |  3 |
| Deep Elf         | -20 |  2 |  -1 |  4 |
| Djinni           | -10 |  0 |   1 |  4 |
| Formicid         |   0 |  0 |   1 |  4 |
| Human            |   0 |  0 |   0 |  3 |
| Merfolk          |   0 |  0 |   0 |  3 |
| Barachi          |   0 |  0 |   0 |  3 |
| Vampire          |   0 |  0 |  -1 |  4 |
| Mummy            |   0 |  0 |  -1 |  5 |
| Coglin           |   0 |  0 |   0 |  5 |
| Draconian Black  |  10 |  0 |  -1 |  3 |
| Draconian Green  |  10 |  0 |  -1 |  3 |
| Draconian Grey   |  10 |  0 |  -1 |  3 |
| Draconian Pale   |  10 |  0 |  -1 |  3 |
| Draconian Purple |  10 |  0 |  -1 |  3 |
| Draconian Red    |  10 |  0 |  -1 |  3 |
| Draconian White  |  10 |  0 |  -1 |  3 |
| Draconian Yellow |  10 |  0 |  -1 |  3 |
| Ghoul            |  10 |  0 |   0 |  3 |
| Armataur         |  10 |  0 |  -1 |  3 |
| Gale Centaur     |  10 |  0 |  -1 |  3 |
| Minotaur         |  10 | -1 |  -1 |  3 |
| Troll            |  30 | -1 |  -1 |  3 |
| Naga             |  20 |  0 |   0 |  5 |
| Demigod          |  10 |  2 |  -2 |  4 |
| Oni              |  30 |  0 |   0 |  4 |
"""


def orgtable_to_dataframe(txt):
    lines = txt.strip().split("\n")
    data_lines = [line for line in lines if not line.strip().startswith("|----------")]
    column_names = [
        col.strip() for col in data_lines[0].strip("|").split("|") if col.strip()
    ]
    data = []
    for line in data_lines[1:]:
        row_values = [val.strip() for val in line.strip("|").split("|") if val.strip()]
        data.append(row_values)
    df = pd.DataFrame(data, columns=column_names).set_index("species")
    for col in df:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


_spec_skills = orgtable_to_dataframe(species_skill_aptitude)
_spec_stats = orgtable_to_dataframe(species_stats)

SPECIES = _spec_stats.index
NUM_SPECIES = len(SPECIES)
SKILLS = _spec_skills.columns.difference(["HP%", "MP", "Exp", "WL"])
NUM_SKILLS = len(SKILLS)
UNUSABLE_SKILL = -99
MUTATION_LEVEL = dict(MUT_UNSKILLED=0)


class Player:
    def __init__(self, species):
        self.species = species
        self.hp_modifier = _spec_stats.loc[species, "HP%"] / 100.0
        self.total_experience = 0
        self.skill_cost_level = 1

        self.skills = {
            "Air": 0,
            "Alc": 0,
            "Arm": 0,
            "Axs": 0,
            "Coj": 0,
            "Ddg": 0,
            "Ear": 0,
            "Evo": 0,
            "Fgr": 0,
            "Fgt": 0,
            "Fir": 0,
            "Hex": 0,
            "Ice": 0,
            "Inv": 0,
            "LBl": 0,
            "MF": 0,
            "Nec": 0,
            "Pla": 0,
            "Rng": 0,
            "SBl": 0,
            "Shd": 0,
            "Shp": 0,
            "Spc": 0,
            "Sth": 0,
            "Stv": 0,
            "Sum": 0,
            "Thr": 0,
            "Trl": 0,
            "UC": 0,
        }

        self.training_targets = self.skills.copy()

        self.skill_points = self.skills.copy()

        self.skill_manual_points = self.skills.copy()

    def has_mutation(self, mut):
        if mut == "MUT_DISTRIBUTED_TRAINING":
            if you.species in ("Gnoll", "Djinni"):
                return True
            else:
                return False

    def wearing_jewellery(self, item):
        # I'm not handling equipment at the moment. FIX ME
        if item == "AMU_WILDSHAPE":
            return False
        return False


you = Player("Human")


class SkillDiff:
    def __init__(self, skill_points=0, experience=0):
        self.skill_points = skill_points
        self.experience = experience

    def __str__(self):
        return f"({self.skill_points}, {self.experience})"


def ASSERT_RANGE(x, xmin, xmax):
    assert not x < xmin, f"{x} < xmin={xmin}"
    assert not x >= xmax, f"{x} >= xmax={xmax}"


def calc_skill_cost(skill_cost_level):
    cost = np.array(
        [
            1,
            2,
            3,
            4,
            5,  # 1-5
            7,
            8,
            9,
            13,
            22,  # 6-10
            37,
            48,
            73,
            98,
            125,  # 11-15
            145,
            170,
            190,
            212,
            225,  # 16-20
            240,
            255,
            260,
            265,
            265,  # 21-25
            265,
            265,
        ]
    )
    assert len(cost) == MAX_SKILL_COST_LEVEL
    ASSERT_RANGE(skill_cost_level, 1, MAX_SKILL_COST_LEVEL + 1)
    return cost[skill_cost_level - 1]


def skill_cost_needed(level):
    return exp_needed(level, 1) * 13


def skill_level_to_diffs(skill, amount, scaled_training=100, base_only=True):
    level = int(amount)
    fractional = amount - level
    if level >= MAX_SKILL_LEVEL:
        level = MAX_SKILL_LEVEL
        fractional = 0
    target = skill_exp_needed(level, skill)

    if fractional:
        target += int(
            (skill_exp_needed(level + 1, skill) - skill_exp_needed(level, skill))
            * fractional
            + 1
        )

    # We're calculating you.skill_points[skill] and calculating the new
    # you.total_experience to update skill cost.

    you_skill = you.skill_points[skill]

    if not base_only:
        # Warning: I haven't implemented this part yet

        # Factor in crosstraining bonus at the time of the query.
        # This will not address the case where some cross-training skills are
        # also being trained.
        you_skill += get_crosstrain_points(skill)

        # Estimate the ash bonus, based on current skill levels and piety.
        # This isn't perfectly accurate, because the boost changes as
        # skill increases. TODO: exact solution.
        # It also assumes that piety won't change.
        if ash_has_skill_boost(skill):
            you_skill += ash_skill_point_boost(skill, you.skills[skill] * 10)

        # Factor in wildshape amulet bonus (+5 shapeshifting levels)
        if skill == "Shp" and you.wearing_jewellery("AMU_WILDSHAPE"):

            wildshape_level = min(you.skills[skill] + 5, MAX_SKILL_LEVEL)
            you_skill += skill_exp_needed(wildshape_level, skill) - skill_exp_needed(
                you.skills[skill], skill
            )

        if you.skill_manual_points[skill]:
            target = int(you_skill + (target - you_skill) // 2)

    if target == you_skill:
        return SkillDiff()

    # Do we need to increase or decrease skill points/xp?
    # XXX: reducing with ash bonuses in play could lead to weird results.
    decrease_skill = target < you_skill

    you_xp = you.total_experience
    you_skill_cost_level = you.skill_cost_level

    if DEBUG:
        print(f"target skill points: {target}")

    while you_skill != target:

        # each loop is the max skill points that can be gained at the
        # current skill cost level, up to `target`.

        # If we are decreasing, find the xp needed to get to the current skill
        # cost level. Otherwise, find the xp needed to get to the next one.
        next_level = skill_cost_needed(
            you_skill_cost_level + (0 if decrease_skill else 1)
        )

        # max xp that can be added (or subtracted) in one pass of the loop
        max_xp = abs(next_level - you_xp)

        # When reducing, we don't want to stop right at the limit, unless
        # we're at skill cost level 0.
        if decrease_skill and you_skill_cost_level:
            max_xp += 1

        cost = calc_skill_cost(you_skill_cost_level)
        # Maximum number of skill points to transfer in one go.
        # It's max_xp/cost rounded up.
        max_skp = max(int((max_xp + cost - 1) / cost), 1)

        delta = SkillDiff()
        delta.skill_points = min(abs(int(target - you_skill)), max_skp)
        delta.experience = int(delta.skill_points * cost)

        if decrease_skill:

            # We are decreasing skill points / xp to reach the target. Ensure
            # that the delta is negative but won't result in negative skp or xp
            delta.skill_points = -min(delta.skill_points, you_skill)
            delta.experience = -min(delta.experience, you_xp)

        if DEBUG:
            print(
                f"cost level: {you_skill_cost_level}, total experience: {you_xp}, "
                f"next level: {next_level}, skill points: {you_skill}, "
                f"delta_skp: {delta.skill_points}, delta_xp: {delta.experience}."
            )

        you_skill += int(
            (delta.skill_points * scaled_training + (-99 if decrease_skill else 99))
            / 100
        )
        you_xp += int(delta.experience)
        you_skill_cost_level = calc_skill_cost_level(you_xp, you_skill_cost_level)

    return SkillDiff(you_skill - you.skill_points[skill], you_xp - you.total_experience)


def calc_skill_cost_level(xp, start):

    while start < MAX_SKILL_COST_LEVEL and xp >= skill_cost_needed(start + 1):

        start += 1

    while start > 0 and xp < skill_cost_needed(start):

        start -= 1

    return start


def get_crosstrain_points(sk):
    points = 0
    for cross in get_crosstrain_skills(sk):
        points += you.skill_points[cross] * 2 / 5
    return points


def get_crosstrain_skills(sk):
    if you.has_mutation("MUT_DISTRIBUTED_TRAINING"):
        return {}
    result = dict(
        SK_SHORT_BLADES=["SK_LONG_BLADES"],
        SK_LONG_BLADES=["SK_SHORT_BLADES"],
        SK_AXES=["SK_POLEARMS", "SK_MACES_FLAILS"],
        SK_STAVES=["SK_POLEARMS", "SK_MACES_FLAILS"],
        SK_MACES_FLAILS=["SK_AXES", "SK_STAVES"],
        SK_POLEARMS=["SK_AXES", "SK_STAVES"],
    )
    return result.get(sk, [])


def skill_exp_needed(lev, sk, sp=None):
    if sp is None:
        sp = you.species
    ASSERT_RANGE(lev, 0, MAX_SKILL_LEVEL + 1)
    return int(_get_skill_cost_for(lev) * species_apt_factor(sk, sp))


@lru_cache(maxsize=1)
def _get_skill_cost_table():
    skill_cost_table = [0] * (MAX_SKILL_LEVEL + 1)
    breakpoints = [9, 18, 26]
    for skill_level in range(MAX_SKILL_LEVEL + 1):
        skill_cost_table[skill_level] = _modulo_skill_cost(skill_level)
        for break_idx in range(len(breakpoints)):
            breakpoint = breakpoints[break_idx]
            if skill_level <= breakpoint:
                break
            skill_cost_table[skill_level] += (
                _modulo_skill_cost(skill_level - breakpoint) / 2
            )
    return skill_cost_table


def _modulo_skill_cost(modulo_level):
    return 25 * modulo_level * (modulo_level + 1)


def _get_skill_cost_for(level):
    skill_cost_table = _get_skill_cost_table()
    return int(skill_cost_table[level])


def species_apt_factor(sk, sp):
    return apt_to_factor(species_apt(sk, sp))


def species_apt(skill, species):
    return max(
        UNUSABLE_SKILL,
        _spec_skills.loc[species, skill] - MUTATION_LEVEL["MUT_UNSKILLED"],
    )


def apt_to_factor(apt):
    return 1 / math.exp(math.log(2) * apt / APT_DOUBLE)


def apt_to_factor(apt):
    return 1 / math.exp(math.log(2) * apt / APT_DOUBLE)


def exp_needed(lev, exp_apt):
    """
    @param lev          The XL to reach.
    @param exp_apt      The XP aptitude to use. If -99, use the current species'.
    @return     The total number of XP points needed to get to the given XL.
    """
    level = (
        1
        if lev == 1
        else 10
        if lev == 2
        else 30
        if lev == 3
        else 70
        if lev == 4
        else None
    )
    if level is None:
        if lev < 13:
            lev -= 4
            level = 10 + 10 * lev + (60 << lev)
        else:
            lev -= 12
            level = 16675 + 5985 * lev + 4235 * lev * lev
    return (level - 1) * apt_to_factor(exp_apt - 1)


def set_skill_level(skill, amount, quiet=True, base_only=False):
    level = int(amount)
    diffs = skill_level_to_diffs(skill, amount, base_only=base_only)

    you.skills[skill] = level
    you.skill_points[skill] += diffs.skill_points
    you.total_experience += diffs.experience
    if not quiet:
        print(
            f"Change (total): {diffs.skill_points} skp ({you.skill_points[skill]}), {diffs.experience} xp ({you.total_experience})"
        )
    check_skill_cost_change(quiet)


def check_skill_cost_change(quiet):
    """
    This *changes* you.skill_cost_level to the right skill_cost_level
    as given by you.total_experience
    """
    initial_cost = you.skill_cost_level
    you.skill_cost_level = calc_skill_cost_level(
        you.total_experience, you.skill_cost_level
    )
    if not quiet and initial_cost != you.skill_cost_level:
        print(f"Adjusting skill cost level to {you.skill_cost_level}")


def ash_has_skill_boost(sk):
    # TODO: implement me!
    return False
