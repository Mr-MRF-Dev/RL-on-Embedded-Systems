"""Tabular Q-Learning on the DPM environment (PC reference implementation).

Trains an agent, then compares it against two fixed baselines:
  - Always-Active
  - Threshold: Sleep when the battery is critical/low, otherwise Active

Usage:
    python train_qlearning.py
"""

import random
from statistics import mean, pstdev

from env import DPMEnv

# Hyperparameters (kept identical to the Arduino sketch)
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 2500
N_EPISODES = 6000

# Evaluation
N_EVAL_EPISODES = 200
N_EVAL_EPISODES_TRAINED = 100

ENV_SEED = 42
AGENT_SEED = 1

ACTION_LETTER = {0: "A", 1: "I", 2: "S"}


def epsilon_at(episode):
    frac = min(1.0, episode / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


def greedy_action(q_table, state):
    return max(range(DPMEnv.N_ACTIONS), key=lambda a: q_table[state][a])


def train(env, rng):
    q_table = [[0.0] * DPMEnv.N_ACTIONS for _ in range(DPMEnv.N_STATES)]
    returns, survivals = [], []

    for episode in range(N_EPISODES):
        state = env.reset()
        eps = epsilon_at(episode)
        total_reward = 0.0

        while True:
            if rng.random() < eps:
                action = rng.randrange(DPMEnv.N_ACTIONS)
            else:
                action = greedy_action(q_table, state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            target = reward if terminated else reward + GAMMA * max(q_table[next_state])
            q_table[state][action] += ALPHA * (target - q_table[state][action])

            state = next_state
            total_reward += reward
            if terminated or truncated:
                break

        returns.append(total_reward)
        survivals.append(env.t)

    return q_table, returns, survivals


def evaluate(env, policy, n_episodes):
    """Run a fixed policy; return (mean_return, std_return, mean_survival, std_survival)."""
    returns, survivals = [], []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        while True:
            state, reward, terminated, truncated, _ = env.step(policy(state))
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
        survivals.append(env.t)
    return mean(returns), pstdev(returns), mean(survivals), pstdev(survivals)


def print_results(rows):
    print("--- evaluation (greedy) ---")
    print(f"{'policy':<22}{'avg_return':>12}{'std_return':>12}"
          f"{'avg_survival':>14}{'std_survival':>14}")
    for name, (ret, std_ret, surv, std_surv) in rows:
        print(f"{name:<22}{ret:>12.2f}{std_ret:>12.2f}{surv:>14.1f}{std_surv:>14.2f}")


def print_policy(q_table):
    print("\nlearned greedy policy (battery_bin, idle_bin) -> action:")
    for b in range(DPMEnv.N_BATTERY_BINS):
        row = [ACTION_LETTER[greedy_action(q_table, b * DPMEnv.N_IDLE_BINS + i)]
               for i in range(DPMEnv.N_IDLE_BINS)]
        print(f"  battery_bin={b}: {row}")


def main():
    env = DPMEnv(seed=ENV_SEED)
    rng = random.Random(AGENT_SEED)

    q_table, returns, survivals = train(env, rng)

    print("--- last 100 training episodes ---")
    print(f"avg return: {mean(returns[-100:]):.2f}   "
          f"avg survival steps: {mean(survivals[-100:]):.1f}\n")

    always_active = evaluate(env, lambda s: 0, N_EVAL_EPISODES)
    threshold = evaluate(
        env,
        lambda s: 0 if (s // DPMEnv.N_IDLE_BINS) <= 1 else 2,
        N_EVAL_EPISODES,
    )
    trained = evaluate(env, lambda s: greedy_action(q_table, s), N_EVAL_EPISODES_TRAINED)

    print_results([
        ("Always-Active", always_active),
        ("Threshold baseline", threshold),
        ("Trained Tabular Q", trained),
    ])
    print_policy(q_table)


if __name__ == "__main__":
    main()
