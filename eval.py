import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from hammurabi_env import DemocraticHammurabi
from train import DiscreteHammurabiEnv, normalize_obs, decode_action_str
from model import DreamerV3WorldModel, DreamerV3Actor, DreamerV3Critic

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def evaluate(episodes: int = 10, ckpt_prefix: str = "hammurabi_coinage_final"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f" EVALUATING DREAMERV3 ON HAMMURABI COINAGE ({device})")
    print("=" * 65)

    OBS_DIM = 11
    ACTION_DIM = 81
    STATE_DIM = 256
    BELIEF_DIM = 512
    NUM_BINS = 255

    env = DiscreteHammurabiEnv(max_years=12)

    # Initialize Networks
    world_model = DreamerV3WorldModel(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        embed_dim=BELIEF_DIM,
        hidden_dim=BELIEF_DIM,
        num_categoricals=16,
        num_classes=16,
        num_reward_bins=NUM_BINS,
        device=device,
    ).to(device)

    actor = DreamerV3Actor(
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        belief_dim=BELIEF_DIM,
        hidden_dim=BELIEF_DIM,
        device=device,
    ).to(device)

    # Load weights
    wm_path = f"{ckpt_prefix}_wm.safetensors"
    actor_path = f"{ckpt_prefix}_actor.safetensors"

    if os.path.exists(wm_path) and os.path.exists(actor_path):
        print(f"Loading weights from {ckpt_prefix}_*.safetensors...")
        world_model.load_state_dict(load_file(wm_path))
        actor.load_state_dict(load_file(actor_path))
    else:
        print(f"Error: Could not find checkpoint files with prefix '{ckpt_prefix}'!")
        return

    world_model.eval()
    actor.eval()

    total_returns = []
    years_reached = []
    final_pops = []
    final_silvers = []
    final_scores = []
    reasons = []
    action_counts = {}

    term_completions = 0
    election_1_passed = 0
    election_2_passed = 0

    print(f"\nRunning {episodes} evaluation episodes...\n")

    for ep in range(episodes):
        obs, info = env.reset()
        state, belief = world_model.init_state(batch_size=1, device=device)
        done = False
        ep_ret = 0.0

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                obs_embed = world_model.encoder(obs_t)
                _, state = world_model.rssm_posterior(belief, obs_embed)

                act_logits = actor(state, belief)
                action = torch.argmax(act_logits, dim=-1).item()

            action_counts[action] = action_counts.get(action, 0) + 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward

            if not done:
                obs = next_obs
                act_onehot = F.one_hot(torch.tensor([action], device=device), num_classes=ACTION_DIM).float()
                _, _, belief = world_model.rssm_prior(state, belief, act_onehot)

        # End of episode stats
        yr = info["year"]
        pop = info["population"]
        silv = info["silver"]
        reason = info.get("reason", "Finished")

        years_reached.append(yr)
        final_pops.append(pop)
        final_silvers.append(silv)
        total_returns.append(ep_ret)
        reasons.append(reason)

        if yr >= 5:
            election_1_passed += 1
        if yr >= 9:
            election_2_passed += 1
        if yr >= 12 and "Completed" in reason:
            term_completions += 1

        print(f" Episode {ep + 1:2d}/{episodes}: Reached Year {yr:2d}/12 | "
              f"Pop: {int(pop):3d} | Silver: {int(silv):5d} | "
              f"Return: {ep_ret:>+6.2f} | Reason: {reason}")

    # Overall Summary
    print("\n" + "=" * 65)
    print(" === EVALUATION SUMMARY ===")
    print("=" * 65)
    print(f" Episodes Evaluated:        {episodes}")
    print(f" Full Term Completions:     {term_completions}/{episodes} ({term_completions / episodes * 100:.1f}%)")
    print(f" Passed Election 1 (Yr 4):  {election_1_passed}/{episodes} ({election_1_passed / episodes * 100:.1f}%)")
    print(f" Passed Election 2 (Yr 8):  {election_2_passed}/{episodes} ({election_2_passed / episodes * 100:.1f}%)")
    print(f" Average Year Reached:      {np.mean(years_reached):.1f} / 12 (Max: Year {max(years_reached)})")
    print(f" Average Population:        {np.mean(final_pops):.0f} citizens")
    print(f" Average Silver Vault:      {np.mean(final_silvers):.0f} shekels")
    print(f" Average Episode Return:    {np.mean(total_returns):+.2f}")
    print("-" * 65)

    print("\nTop 5 Most Frequent AI Actions:")
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    total_acts = sum(action_counts.values())
    for act_idx, count in sorted_actions:
        pct = count / total_acts * 100
        desc = decode_action_str(act_idx)
        print(f"  - Action {act_idx:2d} ({desc:<45}): {count:3d} ({pct:4.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()
