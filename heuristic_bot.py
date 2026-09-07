import os
import time
import numpy as np
from hammurabi_env import DemocraticHammurabi
from train import normalize_obs

def heuristic_policy(env: DemocraticHammurabi) -> list:
    """
    Optimal JIT (Just-In-Time) Economic Policy designed by the User:
    1. Calculate exact annual operational need: (num_worker * 20) + (plant_grain_need).
    2. If grain > total_need: Export exact surplus to rat-immune silver vault.
    3. If grain < total_need: Import exact deficit from silver vault.
    4. Feed 100% of workers and plant 100% of workable capacity.
    5. Opportunistically buy land when cheap if silver reserves allow.
    """
    pop = env.population
    grain = env.grain
    land = env.land
    silver = env.silver
    l_price = env.land_price
    g_price = env.grain_price

    elite_pop = getattr(env, "elite_pop", int(pop * 0.05))
    farmer_pop = getattr(env, "farmer_pop", int(pop * 0.80))
    worker_pop = getattr(env, "worker_pop", pop - elite_pop - farmer_pop)

    max_workable_land = farmer_pop * 10

    # -------------------------------------------------------------------------
    # 1. Real Estate Strategy (Buy dips if silver reserve has ample buffer)
    # -------------------------------------------------------------------------
    action_land = 0.0
    safe_silver_buffer = 2000.0  # Keep emergency silver for grain/calamities

    if l_price <= 23.0 and land < max_workable_land and silver > safe_silver_buffer:
        excess_silver = silver - safe_silver_buffer
        max_acres_affordable = int(excess_silver // l_price)
        target_buy = min(max_acres_affordable, max_workable_land - land, 200)
        if target_buy > 0:
            max_buy_possible = int(silver // l_price)
            action_land = target_buy / max(1, max_buy_possible)

    # Calculate virtual land & silver after land trade
    if action_land < 0:
        acres_sold = int(abs(action_land) * land)
        v_land = land - acres_sold
        v_silver = silver + (acres_sold * l_price)
    elif action_land > 0:
        max_buy = int(silver // l_price)
        acres_bought = int(action_land * max_buy)
        v_land = land + acres_bought
        v_silver = silver - (acres_bought * l_price)
    else:
        v_land = land
        v_silver = silver

    # -------------------------------------------------------------------------
    # 2. Just-In-Time (JIT) Grain Merchant Exchange
    # -------------------------------------------------------------------------
    food_needed = worker_pop * 20
    plant_needed = min(int(v_land), farmer_pop * 10)
    total_need = food_needed + plant_needed

    action_grain_trade = 0.0

    if grain > total_need:
        # SURPLUS: Export 100% of excess grain to rat-immune silver!
        surplus = grain - total_need
        action_grain_trade = -(float(surplus) / max(1.0, float(grain)))
        v_grain = grain - surplus
        v_silver += int(surplus * g_price)

    elif grain < total_need:
        # DEFICIT: Import exact necessary bushels using silver vault!
        deficit = total_need - grain
        max_import_affordable = int(v_silver // g_price)
        bushels_to_buy = min(deficit, max_import_affordable)
        if max_import_affordable > 0 and bushels_to_buy > 0:
            action_grain_trade = float(bushels_to_buy) / float(max_import_affordable)
            v_grain = grain + bushels_to_buy
            v_silver -= int(bushels_to_buy * g_price)
        else:
            v_grain = grain
    else:
        v_grain = grain

    # -------------------------------------------------------------------------
    # 3. Feed Workers (100% Full Bellies)
    # -------------------------------------------------------------------------
    target_food = min(v_grain, food_needed)
    action_feed = float(target_food) / max(1.0, float(v_grain))
    grain_after_food = max(0, v_grain - target_food)

    # -------------------------------------------------------------------------
    # 4. Plant Seeds (100% Workable Land)
    # -------------------------------------------------------------------------
    target_plant = min(int(v_land), farmer_pop * 10, grain_after_food)
    action_plant = float(target_plant) / max(1.0, float(grain_after_food))

    return [action_land, action_grain_trade, action_feed, action_plant]


def run_heuristic_bot(episodes: int = 10, save_demonstrations: bool = True):
    print("=" * 70)
    print(" RUNNING JIT OPTIMAL HEURISTIC BOT (USER ECONOMIC STRATEGY)")
    print("=" * 70)

    env = DemocraticHammurabi(max_years=12)

    final_scores = []
    final_silvers = []
    final_grains = []
    final_lands = []
    final_pops = []
    years_completed = []
    all_saved_steps = 0

    save_dir = "./human_recordings"
    if save_demonstrations:
        os.makedirs(save_dir, exist_ok=True)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        prev_score = env._calculate_reward()

        traj_obs = []
        traj_act = []
        traj_rew = []
        traj_nw = []

        while not done:
            norm_state = normalize_obs(env._get_state())
            actions = heuristic_policy(env)

            next_obs, total_score, done, info = env.step(actions)

            step_reward = (total_score - prev_score) / 50.0
            if done and env.year >= 12 and "Completed" in info.get("reason", ""):
                step_reward += 5.0
            prev_score = total_score

            traj_obs.append(norm_state)
            traj_act.append(actions)
            traj_rew.append(step_reward)
            traj_nw.append(total_score)

        final_scores.append(total_score)
        final_silvers.append(env.silver)
        final_grains.append(env.grain)
        final_lands.append(env.land)
        final_pops.append(env.population)
        years_completed.append(env.year)

        print(f" Episode {ep + 1:2d}/{episodes}: Reached Year {env.year:2d}/12 | "
              f"Pop: {env.population:3d} | Grain: {env.grain:5d} | Silver: {int(env.silver):5d} | "
              f"Land: {env.land:4d} | Score: {total_score:6.1f} | Reason: {info.get('reason', 'Term finished')}")

        if save_demonstrations and env.year >= 12:
            traj_obs.append(normalize_obs(env._get_state()))
            timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{ep}"
            save_path = os.path.join(save_dir, f"hammurabi_coinage_human_{timestamp}.npz")
            np.savez_compressed(
                save_path,
                observations=np.array(traj_obs, dtype=np.float32),
                actions=np.array(traj_act, dtype=np.float32),
                rewards=np.array(traj_rew, dtype=np.float32),
                net_worth=np.array(traj_nw, dtype=np.float32),
            )
            all_saved_steps += len(traj_act)

    print("\n" + "=" * 70)
    print(" === HEURISTIC BOT BENCHMARK SUMMARY ===")
    print("=" * 70)
    print(f" Episodes Evaluated:        {episodes}")
    print(f" Full Term Completions:     {sum(1 for y in years_completed if y >= 12)}/{episodes} ({sum(1 for y in years_completed if y >= 12)/episodes*100:.1f}%)")
    print(f" Average Year Reached:      {np.mean(years_completed):.1f} / 12")
    print(f" Average Final Population:  {np.mean(final_pops):.0f} citizens")
    print(f" Average Final Grain Silos: {np.mean(final_grains):.0f} bushels")
    print(f" Average Final Silver Vault:{np.mean(final_silvers):.0f} shekels")
    print(f" Average Final Land Owned:  {np.mean(final_lands):.0f} acres")
    print(f" Average Final Game Score:  {np.mean(final_scores):.1f}")
    if save_demonstrations:
        print(f" Saved {all_saved_steps} optimal demonstration steps to '{save_dir}/'")
    print("=" * 70)


if __name__ == "__main__":
    run_heuristic_bot(episodes=50, save_demonstrations=True)
