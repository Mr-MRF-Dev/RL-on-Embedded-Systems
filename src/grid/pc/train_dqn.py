"""DQN on Grid-World (PC reference implementation, PyTorch).

The observation is the normalized (row, col) position, so the network size does
not depend on the grid size (unlike a Q-table). The best checkpoint seen during
training is evaluated greedily and its weights are saved to
weights/grid_dqn_n<GRID_N>.pt for export to the microcontroller
(see tools/export_weights_to_c.py).

The same unmodified script can be run on a Raspberry Pi as a control group.

Presets (set GRID_PRESET):
    1  GRID_N = 10,  3000 episodes   quick sanity run
    2  GRID_N = 30, 30000 episodes   large grid; the fixed 3000-episode budget
                                     does not reach the goal at this size

Usage:
    pip install torch numpy
    cd pc
    python train_dqn.py
"""

import copy
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import GridEnv

GRID_PRESET = 1

if GRID_PRESET == 1:
    GRID_N = 10
    N_EPISODES = 3000
    EPS_DECAY_EPISODES = 1500
    LOG_INTERVAL = 100
    PROBE_INTERVAL = 100
else:
    GRID_N = 30
    N_EPISODES = 30000
    EPS_DECAY_EPISODES = 15000
    LOG_INTERVAL = 250
    PROBE_INTERVAL = 250

SEED = 0
N_EVAL_EPISODES = 100

# Checkpoint selection: the last weights are not necessarily the best ones,
# so a short greedy probe runs periodically and the best network is kept.
PROBE_EPISODES = 10

GAMMA = 0.95
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 20000
MIN_REPLAY_BEFORE_TRAIN = 1000
TARGET_UPDATE_EVERY_STEPS = 500

EPS_START = 1.0
EPS_END = 0.05

# Deliberately small so the network can be exported to a microcontroller
HIDDEN_SIZES = (16, 16)

OBS_DIM = 2
WEIGHTS_PATH = Path(__file__).parent / "weights" / f"grid_dqn_n{GRID_N}.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_continuous_state(env):
    return np.array([env.row / (env.n - 1), env.col / (env.n - 1)], dtype=np.float32)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


def epsilon_at(episode):
    frac = min(1.0, episode / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


def greedy_action(policy_net, state):
    with torch.no_grad():
        q = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())


def probe_policy(env, policy_net, n_episodes):
    """Short greedy run used only to select the best checkpoint."""
    policy_net.eval()
    scores = []
    for _ in range(n_episodes):
        env.reset()
        state = get_continuous_state(env)
        total = 0.0
        while True:
            _, reward, terminated, truncated, _ = env.step(
                greedy_action(policy_net, state)
            )
            state = get_continuous_state(env)
            total += reward
            if terminated or truncated:
                break
        scores.append(total)
    policy_net.train()
    return sum(scores) / len(scores)


def optimize(policy_net, target_net, optimizer, replay):
    b_s, b_a, b_r, b_s2, b_done = replay.sample(BATCH_SIZE)
    q_values = policy_net(b_s).gather(1, b_a)
    with torch.no_grad():
        max_next_q = target_net(b_s2).max(dim=1, keepdim=True)[0]
        td_target = b_r + GAMMA * max_next_q * (1.0 - b_done)
    loss = nn.functional.mse_loss(q_values, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(env):
    """Return (best_policy_net, best_probe_score, training_seconds)."""
    n_actions = env.N_ACTIONS

    policy_net = QNetwork(OBS_DIM, n_actions)
    target_net = QNetwork(OBS_DIM, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_CAPACITY)

    global_step = 0
    returns_log = []
    best_score = float("-inf")
    best_state = copy.deepcopy(policy_net.state_dict())

    start = time.time()
    for episode in range(N_EPISODES):
        env.reset()
        state = get_continuous_state(env)
        eps = epsilon_at(episode)
        total_reward = 0.0

        while True:
            if random.random() < eps:
                action = random.randrange(n_actions)
            else:
                action = greedy_action(policy_net, state)

            _, reward, terminated, truncated, _ = env.step(action)
            next_state = get_continuous_state(env)
            # bootstrap on truncation, not on reaching the goal
            replay.push(state, action, reward, next_state, float(terminated))

            state = next_state
            total_reward += reward
            global_step += 1

            if len(replay) >= MIN_REPLAY_BEFORE_TRAIN:
                optimize(policy_net, target_net, optimizer, replay)
                if global_step % TARGET_UPDATE_EVERY_STEPS == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if terminated or truncated:
                break

        returns_log.append(total_reward)
        if (episode + 1) % LOG_INTERVAL == 0:
            recent = returns_log[-LOG_INTERVAL:]
            print(
                f"episode {episode + 1:>5}  "
                f"avg_return(last {LOG_INTERVAL})={sum(recent) / len(recent):7.2f}  eps={eps:.3f}"
            )

        if (episode + 1) % PROBE_INTERVAL == 0:
            score = probe_policy(env, policy_net, PROBE_EPISODES)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(policy_net.state_dict())
                print(
                    f"  -> new best checkpoint at episode {episode + 1}: "
                    f"greedy probe avg_return={score:.2f}"
                )

    training_seconds = time.time() - start
    policy_net.load_state_dict(best_state)
    return policy_net, best_score, training_seconds


def evaluate(env, policy_net):
    """Greedy evaluation; return (mean_return, std_return, mean_steps,
    goal_reach_rate, avg_inference_us)."""
    policy_net.eval()
    returns, steps = [], []
    goal_reached = 0
    inference_time = 0.0
    inference_count = 0

    for _ in range(N_EVAL_EPISODES):
        env.reset()
        state = get_continuous_state(env)
        total_reward = 0.0
        while True:
            t0 = time.perf_counter()
            action = greedy_action(policy_net, state)
            inference_time += time.perf_counter() - t0
            inference_count += 1

            _, reward, terminated, truncated, _ = env.step(action)
            state = get_continuous_state(env)
            total_reward += reward
            if terminated:
                goal_reached += 1
            if terminated or truncated:
                break
        returns.append(total_reward)
        steps.append(env.t)

    mean_ret = sum(returns) / len(returns)
    std_ret = (sum((x - mean_ret) ** 2 for x in returns) / len(returns)) ** 0.5
    mean_steps = sum(steps) / len(steps)
    return (
        mean_ret,
        std_ret,
        mean_steps,
        goal_reached / N_EVAL_EPISODES,
        inference_time / inference_count * 1e6,
    )


def main():
    set_seed(SEED)
    env = GridEnv(n=GRID_N, seed=SEED)

    policy_net, best_score, training_seconds = train(env)

    print(
        f"\ntotal training wall-clock time: {training_seconds:.1f} s "
        f"({training_seconds / 60:.2f} min) over {N_EPISODES} episodes"
    )
    print(
        f"best checkpoint: avg_return={best_score:.2f} "
        f"(greedy probe over {PROBE_EPISODES} episodes)"
    )

    mean_ret, std_ret, mean_steps, goal_rate, inference_us = evaluate(env, policy_net)
    print(f"\nGRID_N={GRID_N}  optimal_return={env.optimal_return():.1f}")
    print(f"=== final evaluation ({N_EVAL_EPISODES} greedy episodes) ===")
    print(
        f"mean_return={mean_ret:.2f}  std_return={std_ret:.2f}  "
        f"mean_steps={mean_steps:.2f}  goal_reach_rate={goal_rate:.2%}"
    )
    print(f"avg inference latency per step: {inference_us:.2f} us")

    n_params = sum(p.numel() for p in policy_net.parameters())
    print(
        f"policy network parameter count: {n_params}  "
        f"(~{n_params * 4} bytes as float32, independent of GRID_N)"
    )

    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    torch.save(policy_net.state_dict(), WEIGHTS_PATH)
    print(f"weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
