"""Dynamic Power Management (DPM) environment.

State  : battery_bin * N_IDLE_BINS + idle_bin  (20 discrete states)
Actions: 0 = Active, 1 = Idle, 2 = Sleep
"""

import random


class DPMEnv:
    N_BATTERY_BINS = 5  # 0 = critical ... 4 = full
    N_IDLE_BINS = 4  # steps since last request, capped at 3
    N_STATES = N_BATTERY_BINS * N_IDLE_BINS
    N_ACTIONS = 3

    # Energy model (drained from a 0..100 battery)
    POWER_COST = {0: 3.0, 1: 1.0, 2: 0.2}
    BASELINE_LEAK = 0.1

    # Workload model
    REQUEST_PROB = 0.30
    LATENCY_PENALTY = {0: 0.0, 1: 1.0, 2: 3.0}

    # Episode and reward
    MAX_STEPS = 200
    REWARD_SCALE = 3.0
    SURVIVAL_BONUS = 1.5  # reward for every step the device stays alive
    TERMINAL_PENALTY = -50.0  # applied when the battery is depleted

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.battery = 100.0
        self.idle_steps = 0
        self.t = 0
        return self._get_state()

    def step(self, action):
        assert action in (0, 1, 2)
        self.t += 1

        cost = self.POWER_COST[action] + self.BASELINE_LEAK
        self.battery -= cost

        request = self.rng.random() < self.REQUEST_PROB
        latency_penalty = 0.0
        if request:
            latency_penalty = self.LATENCY_PENALTY[action]
            self.idle_steps = 0
        else:
            self.idle_steps += 1

        reward = self.SURVIVAL_BONUS - cost / self.REWARD_SCALE - latency_penalty

        terminated = self.battery <= 0.0
        if terminated:
            reward += self.TERMINAL_PENALTY
        truncated = self.t >= self.MAX_STEPS

        info = {"battery": self.battery, "request": request}
        return self._get_state(), reward, terminated, truncated, info

    def _battery_bin(self):
        level = max(0.0, min(100.0, self.battery))
        return min(int(level // 20), self.N_BATTERY_BINS - 1)

    def _idle_bin(self):
        return min(self.idle_steps, self.N_IDLE_BINS - 1)

    def _get_state(self):
        return self._battery_bin() * self.N_IDLE_BINS + self._idle_bin()
