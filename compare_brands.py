#!/usr/bin/env python
import numpy as np
import pandas as pd
import crawl_tab as CT
import crawl_utils as CU
class TooFewSwings(Exception):
    pass
num_iterations = 20000
num_swings = 100

def sim(attacker, defender):
    DAMAGE = CT.damage_instances(attacker, defender, size=(num_iterations, num_swings))
    POWER = DAMAGE * 8
    BASELINE_DELAY = 10
    MISCAST = np.random.random(DAMAGE.shape) < 1 - defender["chance to cast"]
    # print(f"DEBUG: MISCAST rate {MISCAST.mean():.1%}")
    
    if attacker["brand"] == "antimagic":
        DURATION = np.random.randint(low=0, high=(CU.div_rand_round(POWER, defender["HD"]) + 1), size=POWER.shape) * BASELINE_DELAY
        DURATION = DURATION.cumsum(axis=1)
        DURATION -= (DURATION > 0).astype(int).cumsum(axis=1) - (DURATION > 0).astype(int)
        PROB_FAIL = 1 - (4 * BASELINE_DELAY) / (4 * BASELINE_DELAY + DURATION)
        MISCAST |=  np.random.random(PROB_FAIL.shape) < PROB_FAIL
        # print(f"DEBUG: MISCAST rate after antimagic {MISCAST.mean():.1%}")
        
    TOTAL_DAMAGE = DAMAGE.cumsum(axis=1)

    HP = defender["HP"] - TOTAL_DAMAGE
    if (HP > 0)[:, -1].any():
        raise TooFewSwings("Defender did not die. You need to increase the number of swings")

    # attacker_swings is the total number of swings the attacker makes before killing the defender
    attacker_index = np.argmax(HP <= 0, axis=1) 
    attacker_swings = attacker_index + 1

    # attacker_daauts is the time in daauts it takes to kill the defender
    attacker_daauts = attacker_swings * attacker["weapon_delay"]

    # defender_turns is the daaut timestamps when the defender takes an action (assuming defender is alive)
    defender_turns = np.arange(num_swings)
    # defender_index is the index into the MISCAST array when the defender takes an action (assuming defender is alive)
    defender_index = np.ceil(defender_turns / attacker["weapon_delay"]).astype(int)
    defender_index = defender_index[defender_index < num_swings]
    
    # if the defender is dead, surely it will miscast
    INDEX = np.tile(np.arange(num_swings), (num_iterations, 1))
    MISCAST[INDEX >= attacker_index[:, None]] = True

    # check if the defender casts on its turn
    DEFENDER_FAIL = MISCAST[:, defender_index]
    # defender_casts is the number of casts before the defender dies
    defender_casts = (~DEFENDER_FAIL).astype(int).sum(axis=1)
    return attacker["brand"], attacker["weapon_type"], attacker_daauts, defender_casts

def run():
    # defender is an OoF
    # I believe OoFs only cast 40% of the time, so 60% of the time they "fail".
    # See mon-spell.h for the spell cast frequencies. ((32+32+16)/200 = 0.4)
    defender = {'HP': 150, 'chance to cast': 0.4, 'AC': 20, 'EV': 20, 'SH': 0, 'HD': 30}
    attackers = [
            dict(label="antimagic", DEX=20, weapon_type="eveningstar", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=42, weapon_delay=0.7, brand="antimagic"),
             dict(label="speed", DEX=20, weapon_type="eveningstar", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=42, weapon_delay=0.5, brand="speed"),
             dict(label="heavy", DEX=20, weapon_type="eveningstar", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=75, weapon_delay=1.1, brand="heavy"),
             dict(label="freezing", DEX=20, weapon_type="eveningstar", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=42, weapon_delay=0.7, brand="freezing"),
             dict(label="plain", DEX=20, weapon_type="eveningstar", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=42, weapon_delay=0.7, brand="plain"),
            dict(label="antimagic", DEX=20, weapon_type="demon_whip", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=30, weapon_delay=0.5, brand="antimagic"),
             dict(label="speed", DEX=20, weapon_type="demon_whip", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=30, weapon_delay=0.3, brand="speed"),
             dict(label="heavy", DEX=20, weapon_type="demon_whip", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=53, weapon_delay=0.8, brand="heavy"),
             dict(label="freezing", DEX=20, weapon_type="demon_whip", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=30, weapon_delay=0.5, brand="freezing"),
             dict(label="plain", DEX=20, weapon_type="demon_whip", 
                skills={"Maces & Flails":20, "Fighting":20,}, 
                slaying=9, damage=30, weapon_delay=0.5, brand="plain"),
                ]

    
    result = list()
    for attacker in attackers:
        brand, weapon_type, attacker_daauts, defender_casts = sim(attacker, defender)
        result.append((brand, weapon_type, f"{defender_casts.mean():.1f} ± {defender_casts.std(ddof=1):.1f}", f"{attacker_daauts.mean():.1f} ± {attacker_daauts.std(ddof=1):.1f}"))
    df = pd.DataFrame(result, columns=["brand", "weapon_type", "oof_casts", "player_daauts"])
    print(df.to_markdown())

    
if __name__ == "__main__":
    run()
