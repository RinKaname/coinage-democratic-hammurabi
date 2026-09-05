import os
import sys
import time
import numpy as np
from hammurabi_env import DemocraticHammurabi

NORM_FACTORS_11 = np.array([
    12.0,    # year (1..12)
    100.0,   # population (~100..500)
    1000.0,  # grain (~1000..10000)
    1000.0,  # land (~500..2500)
    25.0,    # land_price (20..30 silver/acre)
    30000.0, # silver (scaled for 30,000 starting shekels)
    1.0,     # grain_price (~0.7..2.0 silver/bu)
    100.0,   # farmers_approval (0..100)
    100.0,   # workers_approval (0..100)
    100.0,   # elites_approval (0..100)
    4.0,     # years_until_election (0..3)
], dtype=np.float32)


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    return (np.array(obs, dtype=np.float32) / NORM_FACTORS_11).astype(np.float32)


def play():
    print("=" * 70)
    print("      DEMOCRATIC HAMMURABI - THE SILVER SHEKEL ECONOMY")
    print("    Rule Babylon for 12 years with a Sovereign Monetary Vault!")
    print("=" * 70)

    env = DemocraticHammurabi(max_years=12)
    obs = env.reset()

    traj_obs = []
    traj_act = []
    traj_rew = []
    traj_nw = []

    prev_score = env._calculate_reward()

    while not env.is_done:
        year = int(env.year)
        pop = int(env.population)
        grain = int(env.grain)
        land = int(env.land)
        silver = int(env.silver)
        l_price = env.land_price
        g_price = env.grain_price

        f_appr = env.farmers_approval
        w_appr = env.workers_approval
        e_appr = env.elites_approval
        avg_appr = env.get_average_approval()
        yrs_to_election = 4 - (year % 4) if (year % 4) != 0 else 0

        print(f"\n" + "-" * 70)
        print(f" [YEAR {year} OF 12]  |  Next Election in: {yrs_to_election} year(s)")
        print(f"-" * 70)
        print(f"  Population:          {pop:,} citizens")
        print(f"  Grain in Silos:      {grain:,} bushels (Rats can eat grain!)")
        print(f"  Royal Silver Vault:  {silver:,} shekels (100% IMMUNE to rats!)")
        print(f"  Land Owned:          {land:,} acres")
        print(f"  Current Markets:     Land: {l_price} silver/acre | Grain: {g_price:.2f} silver/bushel")
        print(f"  Faction Approval:     Farmers: {f_appr:.1f}% | Workers: {w_appr:.1f}% | Elites: {e_appr:.1f}%")
        print(f"  Average Approval:    {avg_appr:.1f}% (Required to win election: >= 45.0%)")
        print(f"-" * 70)

        raw_state = env._get_state()
        norm_state = normalize_obs(raw_state)

        # ----------------------------------------------------------------------
        # Decision 1: Land Market (Traded in Silver Coins)
        # ----------------------------------------------------------------------
        max_land_buy = int(silver // l_price)
        print(f"\n1. REAL ESTATE MARKET (Price: {l_price} silver/acre)")
        print(f"   Max you can buy with Silver: {max_land_buy:,} acres | Owned: {land:,} acres")

        while True:
            land_in = input("   Acres to buy (+) with silver or sell (-) for silver [Enter = 0]: ").strip()
            if not land_in:
                acres_trade = 0
                break
            try:
                acres_trade = int(land_in)
                if acres_trade > 0 and acres_trade > max_land_buy:
                    print(f"   Error: You only have {silver} silver! Max you can afford is {max_land_buy} acres.")
                    continue
                if acres_trade < 0 and abs(acres_trade) > land:
                    print(f"   Error: You only own {land} acres to sell!")
                    continue
                break
            except ValueError:
                print("   Invalid number.")

        if acres_trade < 0:
            action_land = -(abs(acres_trade) / max(1, land))
            silver_after_land = silver + (abs(acres_trade) * l_price)
        elif acres_trade > 0:
            action_land = acres_trade / max(1, max_land_buy)
            silver_after_land = silver - (acres_trade * l_price)
        else:
            action_land = 0.0
            silver_after_land = silver

        # ----------------------------------------------------------------------
        # Decision 2: Grain Merchant Exchange (Buy/Sell Grain for Silver)
        # ----------------------------------------------------------------------
        max_grain_import = int(silver_after_land // g_price)
        print(f"\n2. MERCHANT GRAIN EXCHANGE (Price: {g_price:.2f} silver/bushel)")
        print(f"   Treasury: {silver_after_land:,} silver | Silo Grain: {grain:,} bushels")
        print(f"   Max grain you could import: +{max_grain_import:,} bushels")

        while True:
            grain_in = input("   Import grain (+) with silver OR export grain (-) for silver [Enter = 0]: ").strip()
            if not grain_in:
                grain_trade = 0
                break
            try:
                grain_trade = int(grain_in)
                if grain_trade > 0 and grain_trade > max_grain_import:
                    print(f"   Error: You only have {silver_after_land} silver to import grain!")
                    continue
                if grain_trade < 0 and abs(grain_trade) > grain:
                    print(f"   Error: You only have {grain} bushels in silos to export!")
                    continue
                break
            except ValueError:
                print("   Invalid number.")

        if grain_trade < 0:
            # Export / Sell grain for silver
            action_grain_trade = -(abs(grain_trade) / max(1, grain))
            grain_after_trade = grain - abs(grain_trade)
        elif grain_trade > 0:
            # Import / Buy grain with silver
            action_grain_trade = grain_trade / max(1, max_grain_import)
            grain_after_trade = grain + grain_trade
        else:
            action_grain_trade = 0.0
            grain_after_trade = grain

        # ----------------------------------------------------------------------
        # Decision 3: Feed Workers (from Silo Grain)
        # ----------------------------------------------------------------------
        worker_pop = pop - int(pop * 0.05) - int(pop * 0.80)
        needed_food = worker_pop * 20
        print(f"\n3. FEEDING WORKERS (Silos available: {grain_after_trade:,} bushels)")
        print(f"   Workers to feed: {worker_pop:,} | Needed: {needed_food:,} bushels (20 bu/person)")

        default_feed = min(grain_after_trade, needed_food)
        while True:
            food_in = input(f"   Bushels to feed [Enter = {default_feed:,} full feed]: ").strip()
            if not food_in:
                bushels_food = default_feed
                break
            try:
                bushels_food = int(food_in)
                if bushels_food < 0:
                    print("   Cannot be negative.")
                    continue
                if bushels_food > grain_after_trade:
                    print(f"   Error: Only {grain_after_trade:,} bushels available!")
                    continue
                break
            except ValueError:
                print("   Invalid number.")

        action_feed = bushels_food / max(1, grain_after_trade)
        grain_after_food = grain_after_trade - bushels_food

        # ----------------------------------------------------------------------
        # Decision 4: Planting Seeds (from remaining grain)
        # ----------------------------------------------------------------------
        farmer_pop = int(pop * 0.80)
        land_after_trade = land + acres_trade
        max_workable = farmer_pop * 10
        max_plantable = min(land_after_trade, max_workable, grain_after_food)

        print(f"\n4. PLANTING CROPS (Grain available: {grain_after_food:,} bushels)")
        print(f"   1 bu/acre | Farmer capacity: {max_workable:,} acres | Max plantable: {max_plantable:,} acres")

        while True:
            plant_in = input(f"   Acres to plant [Enter = {max_plantable:,} max plant]: ").strip()
            if not plant_in:
                acres_plant = max_plantable
                break
            try:
                acres_plant = int(plant_in)
                if acres_plant < 0:
                    print("   Cannot be negative.")
                    continue
                if acres_plant > max_plantable:
                    print(f"   Error: Max plantable is {max_plantable:,} acres!")
                    continue
                break
            except ValueError:
                print("   Invalid number.")

        action_plant = acres_plant / max(1, grain_after_food)

        # Step Environment
        actions = [action_land, action_grain_trade, action_feed, action_plant]
        next_obs, total_score, done, info = env.step(actions)

        # Step Reward calculation
        step_reward = (total_score - prev_score) / 50.0
        if done and env.year >= 12 and "Completed" in info.get("reason", ""):
            step_reward += 5.0
        prev_score = total_score

        # Record trajectory
        traj_obs.append(norm_state)
        # Store continuous action vector
        traj_act.append(actions)
        traj_rew.append(step_reward)
        traj_nw.append(total_score)

        # Step Report
        starved = env.starved_total
        rats = info.get("rats_ate", 0)
        yield_harvest = info.get("harvest_yield", 3)
        immigrants = info.get("immigrants", 0)

        print(f"\n>>> YEAR {year} HARVEST & FINANCIAL REPORT:")
        print(f"  [+] Harvest Yield:        {yield_harvest} bushels/acre")
        if rats > 0:
            print(f"  [!] RATS ATTACK:          Rats devoured {rats:,} bushels of grain!")
            print(f"      (Your {env.silver:,} silver in the royal vault was completely untouched!)")
        else:
            print(f"  [+] Rats spared the grain silos this year.")
        
        if starved > 0:
            print(f"  [!] FAMINE DISASTER:      {starved:,} citizens starved to death!")
        else:
            print(f"  [+] All citizens were well fed.")

        if immigrants > 0:
            print(f"  [+] Population Growth:    +{immigrants} new settlers attracted to Babylon!")

        print(f"  [=] Vault Silver:         {env.silver:,} shekels")
        print(f"  [=] Silos Grain:          {env.grain:,} bushels")

        if year % 4 == 0:
            post_appr = env.get_average_approval()
            print(f"\n  ============================================================")
            print(f"   ELECTION RESULTS: Approval = {post_appr:.1f}%")
            if post_appr >= 45.0:
                print(f"   RESULT: VICTORY! Re-elected by the citizens!")
            else:
                print(f"   RESULT: DEFEAT! The coalition collapsed.")
            print(f"  ============================================================")

        if done:
            break

    # ==============================================================================
    # Game Over Summary
    # ==============================================================================
    ep_ret = sum(traj_rew)
    print(f"\n" + "=" * 70)
    print(f"                   GAME OVER")
    print(f"=" * 70)
    print(f" Reason:            {info.get('reason', 'Term finished')}")
    print(f" Final Year Reached: Year {env.year}")
    print(f" Final Population:   {env.population:,} citizens")
    print(f" Final Silver Vault: {env.silver:,} shekels")
    print(f" Final Grain:        {env.grain:,} bushels")
    print(f" Final Land:         {env.land:,} acres")
    print(f" Final Game Score:   {total_score:.1f}")
    print(f" RL Episode Return:  {ep_ret:+.2f}  (Matches Dreamer's 'AvRe'!)")
    print(f"=" * 70)

    # Save to human_recordings
    if len(traj_act) > 0:
        save_prompt = input(f"\nSave this gameplay recording to './human_recordings/' for DreamerV3? (Y/n): ").strip().lower()
        if save_prompt in ("", "y", "yes"):
            save_dir = "./human_recordings"
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"hammurabi_coinage_human_{timestamp}.npz")

            traj_obs.append(normalize_obs(env._get_state()))

            np.savez_compressed(
                save_path,
                observations=np.array(traj_obs),
                actions=np.array(traj_act),
                rewards=np.array(traj_rew),
                net_worth=np.array(traj_nw),
            )
            print(f"Saved {len(traj_act)} steps to: '{save_path}'")
            print("DreamerV3 can now learn your Silver Coinage macroeconomic strategy!")


if __name__ == "__main__":
    play()
