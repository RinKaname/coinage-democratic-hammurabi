import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hammurabi_env import DemocraticHammurabi

class GymDemocraticHammurabi(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    Features the 11-dimensional state and 4 continuous action channels.
    """
    metadata = {'render_modes': ['console']}

    def __init__(self, max_years=12):
        super(GymDemocraticHammurabi, self).__init__()
        self.env = DemocraticHammurabi(max_years=max_years)

        # Action space: 4 continuous values between -1 and 1:
        # [0]: Land trade (silver), [1]: Grain trade (silver), [2]: Feed fraction, [3]: Plant fraction
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: 11 features (includes Silver vault and Grain price)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(11,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        action_land = float(action[0])
        action_grain_trade = float(action[1])
        action_feed = float((action[2] + 1.0) / 2.0)   # Map [-1, 1] to [0, 1]
        action_plant = float((action[3] + 1.0) / 2.0)  # Map [-1, 1] to [0, 1]

        env_actions = [action_land, action_grain_trade, action_feed, action_plant]

        obs, reward, done, info = self.env.step(env_actions)
        terminated = done
        truncated = False

        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        print(self.env._get_state())