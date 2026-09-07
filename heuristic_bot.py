import os
import time
import numpy as np
from hammurabi_env import DemocraticHammurabi
from train import normalize_obs

def heuristic_policy(env: DemocraticHammurabi) -> list:
    """
    Royal Advisor Heuristic Policy:
    1. Never starves state workers.
    2. Imports emergency grain if silos are below food requirements.
    3. Exports excess surplus grain to silver vaults to protect against rat plagues.
    4. Buys land on the dip (price <= 22) if silver reserves allow.
    5. Plants 100% of workable farmer acres to secure the +5% Farmers Approval bonus.
    """
    year = env.year
    pop = env.population
    grain = env.grain
    land = env.land
    silver = env.silver
    l_price = env.land_price
    g_price = env.grain_price
    farmer_pop = env.farmer_pop
    worker_pop = env.worker_pop

    # -------------------------------------------------------------------------
    # 1. Real Estate Strategy (Buy dips, avoid selling unless crisis)
    # -------------------------------------------------------------------------
    action_land = 0.0
    max_workable_land = farmer_pop * 10

    if l_price <= 22 and land < max_workable_land and silver >= 800:
        # Buy the dip! Keep at least 500 silver emergency reserve
        safe_silver = silver - 500
        max_acres = safe_silver // l_price
        target_buy = min(max_acres, max_workable_land - land, 50)
        if target_buy > 0:
            max_buy_affordable = int(silver // l_price)
            action_land = target_buy / max(1, max_buy_affordable)

    elif l_price >= 28 and land > max_workable_land + 50 and silver < 500:
        # Sell excess land at peak price if treasury is running low
        excess_land = land - max_workable_land
        target_sell = min(50, excess_land)
        action_land = -(target_sell / max(1, land))

    # Virtual silver & land after trade
    if action_land < 0:
        v_silver = silver + int(abs(action_land) * land) * l_price
        v_land = land - int(abs(action_land) * land)
    elif action_land > 0:
        max_buy_aff = int(silver // l_price)
        v_silver = silver - int(action_land * max_buy_aff) * l_price
        v_land = land + int(action_land * max_buy_aff)
    else:
        v_silver = silver
        v_land = land

    # -------------------------------------------------------------------------
    # 2. Grain Merchant Strategy (Import on deficit, export on surplus)
    # -------------------------------------------------------------------------
    action_grain_trade = 0.0
    food_needed = worker_pop * 20
    seeds_needed = min(v_land, farmer_pop * 10)
    total_grain_needed = food_needed + seeds_needed

    if grain < total_grain_needed:
        # EMERGENCY: Silos don't even have enough food + seeds! Import with silver!
        deficit = total_grain_needed - grain
        max_import = int(v_silver // g_price)
        import_amount = min(deficit, max_import)
        if max_import > 0 and import_amount > 0:
            action_grain_trade = import_amount / max_import

    elif grain > total_grain_needed + 400 and g_price >= 1.2:
        # SURPLUS: Sell excess grain for silver so rats don't eat it!
        surplus = grain - total_grain_needed
        export_amount = min(int(surplus * 0.6), 500)
        if export_amount > 0:
            action_grain_trade = -(export_amount / max(1, grain))

    # Virtual grain after trade
    if action_grain_trade < 0:
        v_grain = grain - int(abs(action_grain_trade) * grain)
    elif action_grain_trade > 0:
        max_imp = int(v_silver // g_price)
        v_grain = grain + int(action_grain_trade * max_imp)
    else:
        v_grain = grain

    # -------------------------------------------------------------------------
    # 3. Citizen Food Allocation (Target 100% full bellies)
    # -------------------------------------------------------------------------
    target_food = min(v_grain, food_needed)
    action_feed = target_food / max(1.0, float(v_grain))
    grain_after_food = max(0, v_grain - target_food)

    # -------------------------------------------------------------------------
    # 4. Planting Seeds (Plant 100% of workable land)
    # -------------------------------------------------------------------------
    target_plant = min(v_land, farmer_pop * 10, grain_after_food)
    action_plant = target_plant / max(1.0, float(grain_after_food))

    return [float(action_land), float(action_grain_trade), float(action_feed), float(action_plant)]


def run_heuristic_benchmark(episodes: int = 20, save_recordings: bool = True):
    print("=" * 70)
    print(f"   RUNNING ROYAL ADVISOR HEURISTIC BENCHMARK ({episodes} EPISODES)")
    print("=" * 70)

    env = DemocraticHammurabi(max_years=12)

    years_survived = []
    final_pops = []
    final_silvers = []
    final_scores = []
    episode_returns = []
    reasons = []

    election_1_wins = 0
    election_2_wins = 0
    full_term_wins = 0

    if save_recordings:
        save_dir = "./human_recordings"
        os.makedirs(save_dir, exist_ok=True)

    for ep in range(episodes):
        obs = env.reset()
        prev_score = env._calculate_reward()
        done = False
        ep_ret = 0.0

        traj_obs = []
        traj_act = []
        traj_rew = []
        traj_nw = []

        while not done:
            raw_state = env._get_state()
            norm_state = normalize_obs(raw_state)

            actions = heuristic_policy(env)
            next_obs, total_score, done, info = env.step(actions)

            step_reward = (total_score - prev_score) / 50.0
            if done and env.year >= 12 and "Completed" in info.get("reason", ""):
                step_reward += 5.0
            prev_score = total_score
            ep_ret += step_reward

            traj_obs.append(norm_state)
            traj_act.append(actions)
            traj_rew.append(step_reward)
            traj_nw.append(total_score)

        yr = env.year
        pop = env.population
        silv = env.silver
        reason = info.get("reason", "Term finished")

        years_survived.append(yr)
        final_pops.append(pop)
        final_silvers.append(silv)
        final_scores.append(total_score)
        episode_returns.append(ep_ret)
        reasons.append(reason)

        if yr >= 5:
            election_1_wins += 1
        if yr >= 9:
            election_2_wins += 1
        if yr >= 12 and "Completed" in reason:
            full_term_wins += 1

            # Auto-save winning runs to replay buffer!
            if save_recordings:
                traj_obs.append(normalize_obs(env._get_state()))
                timestamp = f"{int(time.time() * 1000)}_{ep}"
                f_path = os.path.join(save_dir, f"hammurabi_coinage_expert_{timestamp}.npz")
                np.savez_compressed(
                    f_path,
                    observations=np.array(traj_obs),
                    actions=np.array(traj_act),
                    rewards=np.array(traj_rew),
                    net_worth=np.array(traj_nw),
                )

        status_tag = "WIN!" if ("Completed" in reason) else "LOST"
        print(f" Episode {ep + 1:2d}/{episodes}: [{status_tag:4s}] Reached Year {yr:2d}/12 | "
              f"Pop: {pop:3d} | Silver: {silv:5d} | "
              f"Score: {total_score:6.1f} | Return: {ep_ret:>+6.2f} | Reason: {reason}")

    # =========================================================================
    # Executive Benchmark Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("             ROYAL ADVISOR HEURISTIC BENCHMARK SUMMARY")
    print("=" * 70)
    print(f" Total Episodes Tested:     {episodes}")
    print(f" Full Term Completions:     {full_term_wins}/{episodes} ({full_term_wins / episodes * 100:.1f}%)")
    print(f" Passed Election 1 (Yr 4):  {election_1_wins}/{episodes} ({election_1_wins / episodes * 100:.1f}%)")
    print(f" Passed Election 2 (Yr 8):  {election_2_wins}/{episodes} ({election_2_wins / episodes * 100:.1f}%)")
    print(f" Average Years Reached:     {np.mean(years_survived):.1f} / 12 years (Max: {max(years_survived)})")
    print(f" Average Final Population:  {np.mean(final_pops):.0f} citizens")
    print(f" Average Final Silver:      {np.mean(final_silvers):.0f} shekels")
    print(f" Average Game Score:        {np.mean(final_scores):.1f} points")
    print(f" Average RL Episode Return: {np.mean(episode_returns):+.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_heuristic_benchmark(episodes=20, save_recordings=True)
