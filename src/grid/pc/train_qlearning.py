"""Tabular Q-Learning on the Grid-World environment (PC reference implementation).

Trains an agent on a small grid and compares it against a random policy.

Usage:
    python train_qlearning.py
"""

import random
from statistics import mean, pstdev

from env import GridEnv

GRID_N = 6

# Hyperparameters (kept identical to the Arduino sketch, core budget)
ALPHA = 0.15
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 800
N_EPISODES = 2000

N_EVAL_EPISODES = 100

ENV_SEED = 42
AGENT_SEED = 1


def epsilon_at(episode):
    frac = min(1.0, episode / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


def greedy_action(q_table, state):
    return max(range(GridEnv.N_ACTIONS), key=lambda a: q_table[state][a])


def train(env, rng):
    q_table = [[0.0] * env.N_ACTIONS for _ in range(env.n_states)]

    for episode in range(N_EPISODES):
        state = env.reset()
        eps = epsilon_at(episode)

        while True:
            if rng.random() < eps:
                action = rng.randrange(env.N_ACTIONS)
            else:
                action = greedy_action(q_table, state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            target = reward if terminated else reward + GAMMA * max(q_table[next_state])
            q_table[state][action] += ALPHA * (target - q_table[state][action])

            state = next_state
            if terminated or truncated:
                break

    return q_table


def evaluate(env, policy, n_episodes):
    """Run a fixed policy; return (mean_return, std_return, mean_steps, goal_reach_rate)."""
    returns, steps, reached = [], [], 0
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        while True:
            state, reward, terminated, truncated, _ = env.step(policy(state))
            total_reward += reward
            if terminated:
                reached += 1
            if terminated or truncated:
                break
        returns.append(total_reward)
        steps.append(env.t)
    return mean(returns), pstdev(returns), mean(steps), reached / n_episodes


def main():
    env = GridEnv(n=GRID_N, seed=ENV_SEED)
    rng = random.Random(AGENT_SEED)

    q_table = train(env, rng)

    random_policy = evaluate(env, lambda s: rng.randrange(env.N_ACTIONS), N_EVAL_EPISODES)
    trained = evaluate(env, lambda s: greedy_action(q_table, s), N_EVAL_EPISODES)

    print(f"grid={GRID_N}x{GRID_N}  n_states={env.n_states}  "
          f"optimal_return={env.optimal_return():.1f}\n")
    print(f"{'policy':<20}{'mean_return':>14}{'std_return':>12}"
          f"{'mean_steps':>12}{'goal_reach_rate':>18}")
    for name, (ret, std_ret, steps, rate) in [
        ("Random", random_policy),
        ("Trained Tabular Q", trained),
    ]:
        print(f"{name:<20}{ret:>14.2f}{std_ret:>12.2f}{steps:>12.1f}{rate:>18.2%}")


if __name__ == "__main__":
    main()
