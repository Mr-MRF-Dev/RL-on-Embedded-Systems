"""Grid-World navigation environment.

The agent starts at (0, 0) and must reach (N-1, N-1) on an N x N grid.

State  : row * N + col
Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
Reward : -1 per step, an extra -2 for bumping into a wall (agent stays in
         place), +50 on reaching the goal (replaces the step cost).
"""

import random


class GridEnv:
    N_ACTIONS = 4

    STEP_COST = -1.0
    WALL_BUMP_PENALTY = -2.0
    GOAL_REWARD = 50.0

    def __init__(self, n=6, max_steps=None, seed=None):
        self.n = n
        self.n_states = n * n
        self.max_steps = max_steps if max_steps is not None else max(50, 6 * n)
        self.goal_row = n - 1
        self.goal_col = n - 1
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.row, self.col = 0, 0
        self.t = 0
        return self._state()

    def step(self, action):
        assert action in (0, 1, 2, 3)
        self.t += 1

        old_row, old_col = self.row, self.col
        if action == 0:
            self.row -= 1
        elif action == 1:
            self.row += 1
        elif action == 2:
            self.col -= 1
        else:
            self.col += 1

        wall_bump = not (0 <= self.row < self.n and 0 <= self.col < self.n)
        if wall_bump:
            self.row, self.col = old_row, old_col

        reward = self.STEP_COST + (self.WALL_BUMP_PENALTY if wall_bump else 0.0)

        terminated = (self.row, self.col) == (self.goal_row, self.goal_col)
        if terminated:
            reward = self.GOAL_REWARD
        truncated = self.t >= self.max_steps

        info = {"row": self.row, "col": self.col}
        return self._state(), reward, terminated, truncated, info

    def optimal_return(self):
        """Best possible return: shortest path is 2*(n-1) steps; every step but
        the last costs -1, the last gives the goal reward."""
        path_len = 2 * (self.n - 1)
        if path_len == 0:
            return self.GOAL_REWARD
        return -(path_len - 1) * 1.0 + self.GOAL_REWARD

    def _state(self):
        return self.row * self.n + self.col
