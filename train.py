import os
import sys
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tqdm import tqdm
from gymnasium import spaces

from hammurabi_env import DemocraticHammurabi
from model import DreamerV3WorldModel, DreamerV3Actor, DreamerV3Critic
from torchrl.objectives.dreamer_v3 import DreamerV3ModelLoss, symlog, symexp, two_hot_encode

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ==============================================================================
# 0. Discrete Environment Wrapper for Democratic Hammurabi (Coinage Economy)
# ==============================================================================
# 81 Discrete Actions = 3 Land Choices x 3 Grain Trade Choices x 3 Feed Choices x 3 Plant Choices
LAND_CHOICES = [-0.5, 0.0, 0.5]         # Sell Land (for Silver), Hold, Buy Land (with Silver)
GRAIN_TRADE_CHOICES = [-0.5, 0.0, 0.5]  # Export Grain (for Silver), Hold, Import Grain (with Silver)
FEED_MULTIPLIERS = [0.85, 1.0, 1.25]    # Rationing (85%), Maintain (100%), Feast (125%)
PLANT_MULTIPLIERS = [0.5, 0.8, 1.0]     # Plant 50%, Plant 80%, Plant 100% full capacity

NORM_FACTORS = np.array([
    12.0,       # year (1..12)
    1000.0,     # population (~100..1000)
    10000.0,    # grain (~1000..10000)
    5000.0,     # land (~500..5000)
    100.0,      # land_price (10..100 silver/acre)
    50000.0,    # silver (~30000..50000 shekels)
    5.0,        # grain_price (~0.5..5.0 silver/bu)
    100.0,      # farmers_approval (0..100)
    100.0,      # workers_approval (0..100)
    100.0,      # elites_approval (0..100)
    4.0,        # years_until_election (0..3)
], dtype=np.float32)


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    return (np.array(obs, dtype=np.float32) / NORM_FACTORS).astype(np.float32)


def decode_action_str(action_idx: int) -> str:
    land_idx = action_idx // 27
    grain_idx = (action_idx // 9) % 3
    feed_idx = (action_idx // 3) % 3
    plant_idx = action_idx % 3

    land_names = ["Sell Land", "Hold Land", "Buy Land"]
    grain_names = ["Export Grain", "Hold Grain", "Import Grain"]
    feed_names = ["Ration (85%)", "Maintain (100%)", "Feast (125%)"]
    plant_names = ["Plant 50%", "Plant 80%", "Plant 100%"]

    return f"{land_names[land_idx]} | {grain_names[grain_idx]} | {feed_names[feed_idx]} | {plant_names[plant_idx]}"


class DiscreteHammurabiEnv:
    """
    Gym wrapper for the 11-feature Silver Shekel Coinage economy.
    Maps 81 strategic discrete actions into balanced continuous execution.
    """
    def __init__(self, max_years: int = 12):
        self.env = DemocraticHammurabi(max_years=max_years)
        self.obs_dim = 11
        self.action_dim = 81
        self.action_space = spaces.Discrete(81)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(11,), dtype=np.float32)
        self.prev_score = 0.0

    def reset(self, seed=None):
        raw_obs = self.env.reset()
        self.prev_score = self.env._calculate_reward()
        norm_obs = normalize_obs(raw_obs)

        info = {
            "year": int(self.env.year),
            "population": float(self.env.population),
            "grain": float(self.env.grain),
            "land": float(self.env.land),
            "silver": float(self.env.silver),
            "avg_approval": float((self.env.farmers_approval + self.env.workers_approval + self.env.elites_approval) / 3.0),
        }
        return norm_obs, info

    def step(self, action_idx: int):
        land_idx = action_idx // 27
        grain_idx = (action_idx // 9) % 3
        feed_idx = (action_idx // 3) % 3
        plant_idx = action_idx % 3

        action_land = LAND_CHOICES[land_idx]
        action_grain_trade = GRAIN_TRADE_CHOICES[grain_idx]

        # 1. Grain available after potential merchant trade
        current_grain = float(self.env.grain)
        if action_grain_trade < 0:
            grain_after_trade = max(0.0, current_grain - int(abs(action_grain_trade) * current_grain))
        elif action_grain_trade > 0:
            max_import = int(self.env.silver // self.env.grain_price)
            grain_after_trade = current_grain + int(action_grain_trade * max_import)
        else:
            grain_after_trade = current_grain

        # 2. Food allocation (Worker Only)
        worker_pop = self.env.population - int(self.env.population * 0.05) - int(self.env.population * 0.80)
        food_needed = worker_pop * 20
        target_food = min(grain_after_trade, food_needed * FEED_MULTIPLIERS[feed_idx])
        action_feed = target_food / max(1.0, grain_after_trade)

        # 3. Planting allocation (calculated on REMAINING grain after food!)
        # Only farmers can plant
        grain_after_food = max(0.0, grain_after_trade - target_food)
        farmer_pop = int(self.env.population * 0.80)
        max_workable = farmer_pop * 10
        target_plant = min(float(self.env.land), float(max_workable)) * PLANT_MULTIPLIERS[plant_idx]
        action_plant = min(1.0, target_plant / max(1.0, grain_after_food))

        actions = [action_land, action_grain_trade, action_feed, action_plant]
        raw_obs, total_score, terminated, info = self.env.step(actions)
        norm_obs = normalize_obs(raw_obs)

        # Dense step reward
        step_reward = (total_score - self.prev_score) / 50.0
        if terminated and self.env.year >= 12 and "Completed" in info.get("reason", ""):
            step_reward += 5.0
        self.prev_score = total_score

        info["year"] = int(self.env.year)
        info["population"] = float(self.env.population)
        info["grain"] = float(self.env.grain)
        info["land"] = float(self.env.land)
        info["silver"] = float(self.env.silver)
        info["avg_approval"] = float((self.env.farmers_approval + self.env.workers_approval + self.env.elites_approval) / 3.0)

        return norm_obs, step_reward, terminated, False, info


# ==============================================================================
# 1. Experience Replay Buffer (Sequence Sampler for World Model Training)
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, obs_dim: int = 11, action_dim: int = 81):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.bool_)

        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def load_recordings(self, folder_path: str = "./human_recordings") -> int:
        """
        Preloads human demonstration recordings (.npz) into the replay buffer.
        """
        import glob
        if not os.path.exists(folder_path):
            return 0

        npz_files = glob.glob(os.path.join(folder_path, "*.npz"))
        if not npz_files:
            return 0

        loaded_steps = 0
        loaded_episodes = 0
        for f in sorted(npz_files):
            try:
                data = np.load(f)
                obs = data['observations'] if 'observations' in data else data['observation']
                act = data['actions'] if 'actions' in data else data['action']
                rew = data['rewards'] if 'rewards' in data else data['reward']

                T = len(rew)
                if len(act) == T + 1:
                    act = act[1:]

                # Check observation dimension compatibility
                if obs.shape[-1] != self.obs_dim:
                    continue

                for t in range(T):
                    is_done = (t == T - 1)
                    # Map continuous human action to discrete 81 action index
                    if isinstance(act[t], (list, np.ndarray)):
                        c_act = act[t]
                        l_val = float(c_act[0])
                        g_val = float(c_act[1])
                        f_val = float(c_act[2])
                        p_val = float(c_act[3])

                        l_idx = int(np.argmin(np.abs(np.array(LAND_CHOICES) - l_val)))
                        g_idx = int(np.argmin(np.abs(np.array(GRAIN_TRADE_CHOICES) - g_val)))
                        f_idx = 2 if f_val > 0.8 else (0 if f_val < 0.6 else 1)
                        p_idx = 0 if p_val < 0.6 else (1 if p_val < 0.9 else 2)

                        act_val = int(l_idx * 27 + g_idx * 9 + f_idx * 3 + p_idx)
                    else:
                        act_val = int(act[t])
                    self.add(obs[t], act_val, float(rew[t]), is_done)
                    loaded_steps += 1
                loaded_episodes += 1
            except Exception as e:
                print(f"Notice: Could not load {f}: {e}")

        if loaded_episodes > 0:
            print(f"Loaded {loaded_episodes} human demonstrations ({loaded_steps} steps) into Replay Buffer!")
        return loaded_steps

    def sample_sequence(self, batch_size: int = 32, seq_len: int = 12, device=None) -> TensorDict:
        required_len = seq_len + 1

        if self.size < self.capacity:
            valid_max = max(1, self.size - required_len)
            start_indices = np.random.randint(0, valid_max, size=batch_size)
        else:
            start_indices = []
            while len(start_indices) < batch_size:
                idx = np.random.randint(0, self.capacity - required_len)
                if not (idx <= self.idx < idx + required_len):
                    start_indices.append(idx)

        seq_obs = np.empty((batch_size, required_len, self.obs_dim), dtype=np.float32)
        seq_act = np.empty((batch_size, required_len), dtype=np.int64)
        seq_rew = np.empty((batch_size, required_len, 1), dtype=np.float32)
        seq_don = np.empty((batch_size, required_len, 1), dtype=np.bool_)

        for b, s_idx in enumerate(start_indices):
            indices = np.arange(s_idx, s_idx + required_len) % self.capacity
            seq_obs[b] = self.obs[indices]
            seq_act[b] = self.actions[indices]
            seq_rew[b] = self.rewards[indices]
            seq_don[b] = self.dones[indices]

        t_obs = torch.tensor(seq_obs[:, :-1], dtype=torch.float32, device=device)
        t_next_obs = torch.tensor(seq_obs[:, 1:], dtype=torch.float32, device=device)
        t_act_idx = torch.tensor(seq_act[:, :-1], dtype=torch.int64, device=device)
        t_act = F.one_hot(t_act_idx, num_classes=self.action_dim).float()
        t_rew = torch.tensor(seq_rew[:, :-1], dtype=torch.float32, device=device)
        t_don = torch.tensor(seq_don[:, :-1], dtype=torch.bool, device=device)

        batch_td = TensorDict({
            "observation": t_obs,
            "action": t_act,
            "state": torch.zeros(*t_obs.shape[:-1], 256, device=device),
            "belief": torch.zeros(*t_obs.shape[:-1], 512, device=device),
            "next": {
                "observation": t_next_obs,
                "reward": t_rew,
                "done": t_don,
            }
        }, batch_size=[batch_size, seq_len], device=device)

        return batch_td


# ==============================================================================
# 2. DreamerV3 Generalized Advantage Estimation / Lambda Returns
# ==============================================================================
def compute_lambda_returns(rewards, continues, next_values, lambda_: float = 0.95, gamma: float = 0.99):
    H, B = rewards.shape[:2]
    returns = torch.zeros_like(next_values)
    last_val = next_values[-1]

    for t in reversed(range(H)):
        returns[t] = rewards[t] + continues[t] * gamma * ((1.0 - lambda_) * next_values[t] + lambda_ * last_val)
        last_val = returns[t]
    return returns


def decode_two_hot(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    symlog_val = torch.sum(probs * bins, dim=-1, keepdim=True)
    return symexp(symlog_val)


# ==============================================================================
# 3. Main Training Pipeline
# ==============================================================================
def train(
    total_steps: int = 50_000,
    prefill_steps: int = 1_000,
    train_every: int = 4,
    batch_size: int = 32,
    seq_len: int = 12,
    imagine_horizon: int = 8,
    resume: bool = True,
    entropy_coef: float = 3e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f" Initializing DreamerV3 on Hammurabi Coinage Economy ({device})")
    print(f"============================================================")

    TOTAL_STEPS = total_steps
    PREFILL_STEPS = prefill_steps
    TRAIN_EVERY = train_every
    BATCH_SIZE = batch_size
    SEQ_LEN = seq_len
    IMAGINE_HORIZON = imagine_horizon
    NUM_BINS = 255
    STATE_DIM = 256
    BELIEF_DIM = 512
    OBS_DIM = 11
    ACTION_DIM = 81

    # Two-hot return bins
    two_hot_bins = torch.linspace(-20.0, 20.0, NUM_BINS, device=device)

    # Initialize Environment
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

    critic = DreamerV3Critic(
        num_value_bins=NUM_BINS,
        state_dim=STATE_DIM,
        belief_dim=BELIEF_DIM,
        hidden_dim=BELIEF_DIM,
        device=device,
    ).to(device)

    # Resume from existing checkpoints if available
    ckpt_prefix = "hammurabi_coinage"
    if resume and os.path.exists(f"{ckpt_prefix}_final_wm.safetensors"):
        print(f"Found existing checkpoints! Loading weights from {ckpt_prefix}_final_*.safetensors...")
        if HAS_SAFETENSORS:
            world_model.load_state_dict(load_file(f"{ckpt_prefix}_final_wm.safetensors"))
            actor.load_state_dict(load_file(f"{ckpt_prefix}_final_actor.safetensors"))
            critic.load_state_dict(load_file(f"{ckpt_prefix}_final_critic.safetensors"))
        else:
            pt_path = f"{ckpt_prefix}_final.pt"
            if os.path.exists(pt_path):
                ckpt = torch.load(pt_path, map_location=device)
                world_model.load_state_dict(ckpt["world_model"])
                actor.load_state_dict(ckpt["actor"])
                critic.load_state_dict(ckpt["critic"])

    # Target Critic
    slow_critic = DreamerV3Critic(
        num_value_bins=NUM_BINS,
        state_dim=STATE_DIM,
        belief_dim=BELIEF_DIM,
        hidden_dim=BELIEF_DIM,
        device=device,
    ).to(device)
    slow_critic.load_state_dict(critic.state_dict())
    for p in slow_critic.parameters():
        p.requires_grad = False

    # Optimizers
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=1e-4, eps=1e-8)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-5, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-5, eps=1e-5)

    # World Model Loss
    wm_loss_fn = DreamerV3ModelLoss(
        world_model,
        lambda_kl=1.0,
        lambda_reco=1.0,
        lambda_reward=1.0,
        num_reward_bins=NUM_BINS,
        free_bits=1.0,
        global_average=True,
    )
    wm_loss_fn.set_keys(pixels="observation", reco_pixels="reco_observation")

    # Replay Buffer
    buffer = ReplayBuffer(capacity=100_000, obs_dim=OBS_DIM, action_dim=ACTION_DIM)

    # Prefill Replay Buffer with human demonstrations if available
    loaded_steps = buffer.load_recordings("./human_recordings")
    if loaded_steps == 0:
        loaded_steps = buffer.load_recordings("./eval_recordings")

    remaining_prefill = max(0, PREFILL_STEPS - loaded_steps)
    if remaining_prefill > 0:
        print(f"Supplementing buffer with {remaining_prefill} exploratory steps...")
        obs, _ = env.reset()
        for _ in tqdm(range(remaining_prefill), desc="Prefill"):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, done)
            obs = env.reset()[0] if done else next_obs

    # Main Training Loop
    print("\nStarting DreamerV3 Coinage Economy Training Loop...")
    pbar = tqdm(range(TOTAL_STEPS), desc="Training Steps")

    obs, info = env.reset()
    state, belief = world_model.init_state(batch_size=1, device=device)

    ep_reward = 0.0
    recent_returns = deque(maxlen=50)
    recent_years = deque(maxlen=50)
    recent_pops = deque(maxlen=50)
    recent_silvers = deque(maxlen=50)
    recent_approvals = deque(maxlen=50)

    wm_loss_val = 0.0
    act_loss_val = 0.0
    val_loss_val = 0.0

    for step in pbar:
        # Action Selection
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_embed = world_model.encoder(obs_t)
            _, state = world_model.rssm_posterior(belief, obs_embed)

            act_logits = actor(state, belief)
            probs = F.softmax(act_logits, dim=-1)
            # 1% uniform noise for exploratory diversity
            probs = 0.99 * probs + 0.01 / ACTION_DIM
            action = torch.distributions.Categorical(probs=probs).sample().item()

        # Step Environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        buffer.add(obs, action, reward, done)

        # Advance recurrent belief
        if done:
            recent_returns.append(ep_reward)
            recent_years.append(info["year"])
            recent_pops.append(info["population"])
            recent_silvers.append(info["silver"])
            recent_approvals.append(info["avg_approval"])

            ep_reward = 0.0
            obs, info = env.reset()
            state, belief = world_model.init_state(batch_size=1, device=device)
        else:
            obs = next_obs
            act_onehot = F.one_hot(torch.tensor([action], device=device), num_classes=ACTION_DIM).float()
            _, _, belief = world_model.rssm_prior(state, belief, act_onehot)

        # Training Step (World Model + Latent Imagination)
        if step % TRAIN_EVERY == 0:
            batch_td = buffer.sample_sequence(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device=device)

            # Train World Model
            wm_opt.zero_grad()
            loss_td, updated_td = wm_loss_fn(batch_td.clone())
            wm_total_loss = (
                loss_td["loss_model_kl"]
                + loss_td["loss_model_reco"]
                + loss_td["loss_model_reward"]
            )
            wm_total_loss.backward()
            nn.utils.clip_grad_norm_(world_model.parameters(), 1000.0)
            wm_opt.step()
            wm_loss_val = wm_total_loss.item()

            # Train Actor & Critic in Imagination
            start_states = updated_td.get(("next", "state")).detach()
            start_beliefs = updated_td.get(("next", "belief")).detach()

            curr_st = start_states.view(-1, STATE_DIM)
            curr_bel = start_beliefs.view(-1, BELIEF_DIM)

            imag_states = [curr_st]
            imag_beliefs = [curr_bel]
            imag_actions = []

            for _ in range(IMAGINE_HORIZON):
                with torch.no_grad():
                    policy_logits = actor(curr_st, curr_bel)
                    policy_dist = torch.distributions.Categorical(logits=policy_logits)
                    act_sampled = policy_dist.sample()
                    act_onehot = F.one_hot(act_sampled, num_classes=ACTION_DIM).float()

                imag_actions.append(act_sampled)
                _, curr_st, curr_bel = world_model.rssm_prior(curr_st, curr_bel, act_onehot)
                imag_states.append(curr_st)
                imag_beliefs.append(curr_bel)

            imag_st = torch.stack(imag_states[1:], dim=0)
            imag_bel = torch.stack(imag_beliefs[1:], dim=0)
            imag_act = torch.stack(imag_actions, dim=0)

            H, N = imag_st.shape[:2]

            with torch.no_grad():
                feat_flat = torch.cat([imag_st.view(H * N, -1), imag_bel.view(H * N, -1)], dim=-1)

                imag_rew_logits = world_model.reward_head(feat_flat).view(H, N, -1)
                imag_rewards = decode_two_hot(imag_rew_logits, two_hot_bins).squeeze(-1)

                imag_cont_logits = world_model.continue_head(feat_flat).view(H, N, -1)
                imag_continues = torch.sigmoid(imag_cont_logits).squeeze(-1)

                slow_val_logits = slow_critic(imag_st.view(H * N, -1), imag_bel.view(H * N, -1)).view(H, N, -1)
                slow_values = decode_two_hot(slow_val_logits, two_hot_bins).squeeze(-1)

                lambda_returns = compute_lambda_returns(imag_rewards, imag_continues, slow_values)

            # Update Actor
            actor_opt.zero_grad()
            curr_policy_logits = actor(imag_st.detach().view(H * N, -1), imag_bel.detach().view(H * N, -1)).view(H, N, -1)
            dist = torch.distributions.Categorical(logits=curr_policy_logits)
            log_probs = dist.log_prob(imag_act)

            with torch.no_grad():
                online_val_logits = critic(imag_st.view(H * N, -1), imag_bel.view(H * N, -1)).view(H, N, -1)
                baseline = decode_two_hot(online_val_logits, two_hot_bins).squeeze(-1)
                advantage = lambda_returns - baseline

                q5, q95 = torch.quantile(advantage, torch.tensor([0.05, 0.95], device=device))
                adv_scale = torch.clamp(q95 - q5, min=1.0)

            actor_loss = -torch.mean(log_probs * (advantage / adv_scale))
            entropy_loss = -entropy_coef * dist.entropy().mean()
            total_act_loss = actor_loss + entropy_loss

            total_act_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
            actor_opt.step()
            act_loss_val = total_act_loss.item()

            # Update Critic
            critic_opt.zero_grad()
            pred_val_logits = critic(imag_st.detach().view(H * N, -1), imag_bel.detach().view(H * N, -1))
            val_targets = two_hot_encode(symlog(lambda_returns.detach().view(-1)), two_hot_bins)
            critic_loss = -torch.sum(val_targets * torch.log_softmax(pred_val_logits, dim=-1), dim=-1).mean()

            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
            critic_opt.step()
            val_loss_val = critic_loss.item()

            # Soft update target critic
            with torch.no_grad():
                for p, p_slow in zip(critic.parameters(), slow_critic.parameters()):
                    p_slow.data.copy_(0.98 * p_slow.data + 0.02 * p.data)

        # Progress Logging
        if step % 100 == 0:
            avg_r = np.mean(recent_returns) if recent_returns else 0.0
            avg_y = np.mean(recent_years) if recent_years else 1.0
            avg_p = np.mean(recent_pops) if recent_pops else 100.0
            avg_s = np.mean(recent_silvers) if recent_silvers else 1000.0
            avg_a = np.mean(recent_approvals) if recent_approvals else 50.0

            pbar.set_postfix({
                "AvRe": f"{avg_r:.2f}",
                "Yr": f"{avg_y:.1f}/12",
                "P": f"{avg_p:.0f}",
                "Sl": f"{avg_s:.0f}",
                "Ap": f"{avg_a:.0f}%",
                "WM": f"{wm_loss_val:.2f}",
                "Act": f"{act_loss_val:.2f}",
                "Val": f"{val_loss_val:.2f}",
            })

        # Save Checkpoints
        if step > 0 and step % 10_000 == 0:
            save_checkpoint(world_model, actor, critic, f"{ckpt_prefix}_step_{step}")

    # Save final model
    save_checkpoint(world_model, actor, critic, f"{ckpt_prefix}_final")
    print(f"\nTraining complete! Checkpoints saved as '{ckpt_prefix}_final_*.safetensors'.")


def save_checkpoint(world_model, actor, critic, prefix: str):
    if HAS_SAFETENSORS:
        save_file(world_model.state_dict(), f"{prefix}_wm.safetensors")
        save_file(actor.state_dict(), f"{prefix}_actor.safetensors")
        save_file(critic.state_dict(), f"{prefix}_critic.safetensors")
    else:
        torch.save({
            "world_model": world_model.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        }, f"{prefix}.pt")


if __name__ == "__main__":
    train()
