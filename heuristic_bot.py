from hammurabi_env import DemocraticHammurabi

def run_heuristic_bot():
    print("======================================================================")
    print("          STARTING HEURISTIC BOT - THE SILVER SHEKEL ECONOMY          ")
    print("======================================================================\n")

    env = DemocraticHammurabi(max_years=12)
    obs = env.reset()

    total_score = 0
    done = False

    while not done:
        year = env.year
        pop = env.population
        grain = env.grain
        silver = env.silver
        land = env.land
        l_price = env.land_price
        g_price = env.grain_price

        farmer_pop = env.farmer_pop
        worker_pop = env.worker_pop

        print(f"--- YEAR {year} ---")
        print(f"Start | Pop: {pop} (Workers: {worker_pop}, Farmers: {farmer_pop}) | Grain: {int(grain)} | Silver: {int(silver)} | Land: {int(land)}")
        print(f"Prices| Land: {l_price:.1f} | Grain: {g_price:.2f}")

        # ---------------------------------------------------------
        # Step 1: Feed Workers
        # ---------------------------------------------------------
        needed_food = worker_pop * 20

        # ---------------------------------------------------------
        # Step 2: Calculate Seed Needs
        # ---------------------------------------------------------
        max_workable_land = farmer_pop * 10
        target_planting = min(land, max_workable_land)

        total_grain_needed = needed_food + target_planting

        # ---------------------------------------------------------
        # Step 3: Trade Logic (Grain & Land)
        # ---------------------------------------------------------
        action_grain_trade = 0.0
        action_land = 0.0

        # Do we need to import grain to survive?
        if grain < total_grain_needed:
            deficit = total_grain_needed - grain
            max_import = int(silver // g_price)
            if max_import > 0:
                bushels_to_buy = min(deficit, max_import)
                action_grain_trade = bushels_to_buy / max_import
                print(f"  -> IMPORTING: {bushels_to_buy} grain to cover deficit.")

        # Do we have a massive surplus we should sell?
        elif grain > total_grain_needed + 2000:  # Keep 2000 as buffer
            surplus = grain - (total_grain_needed + 2000)
            action_grain_trade = -(surplus / max(1, grain))
            print(f"  -> EXPORTING: {int(surplus)} grain for profit.")

        # Simulate grain after trade for the feed/plant fractions
        if action_grain_trade > 0:
            max_import = int(silver // g_price)
            grain_after_trade = grain + int(action_grain_trade * max_import)
        elif action_grain_trade < 0:
            grain_after_trade = grain - int(abs(action_grain_trade) * grain)
        else:
            grain_after_trade = grain

        # Buy land if we have excess silver, but ONLY if we have enough farmers to work it!
        excess_silver = silver - 5000 # Keep 5k emergency fund
        if excess_silver > 0 and land < max_workable_land:
            max_buyable_acres = int(excess_silver // l_price)
            acres_needed = max_workable_land - land

            acres_to_buy = min(max_buyable_acres, acres_needed)
            if acres_to_buy > 0:
                max_possible = int(silver // l_price)
                action_land = acres_to_buy / max(1, max_possible)
                print(f"  -> BUYING: {acres_to_buy} acres of land.")

        # ---------------------------------------------------------
        # Step 4: Execute Feeding and Planting
        # ---------------------------------------------------------
        # Convert exact bushel amounts into the 0.0 to 1.0 continuous actions expected by env

        actual_food_to_give = min(needed_food, grain_after_trade)
        action_feed = actual_food_to_give / max(1, grain_after_trade)

        grain_after_food = grain_after_trade - actual_food_to_give

        actual_seeds_to_plant = min(target_planting, grain_after_food)
        action_plant = actual_seeds_to_plant / max(1, grain_after_food)

        actions = [action_land, action_grain_trade, action_feed, action_plant]

        obs, reward, done, info = env.step(actions)

        print(f"End   | Approval: {env.get_average_approval():.1f}% | Starved: {env.starved_total} | Harvest: {info['harvest_yield']} | Rats ate: {info['rats_ate']}")
        print(f"--------------------------------------------------\n")

    print(f"GAME OVER. Reason: {info['reason']}")
    print(f"Final Score: {reward:.1f} | Final Pop: {env.population} | Final Silver: {int(env.silver)}")

if __name__ == "__main__":
    run_heuristic_bot()
