"""Tabular Q-Learning on the Cart-Pole environment (PC reference implementation).

Trains an agent on a discretized state space and compares it against a
random policy.

Usage:
    python train_qlearning.py
"""

import random
from statistics import mean, pstdev

from env import CartPoleEnv

NUM_BINS = 6

# Hyperparameters (kept identical to the Arduino sketch, core budget)
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 4000
N_EPISODES = 10000

N_EVAL_EPISODES = 100
LOG_INTERVAL = 1000

ENV_SEED = 42
AGENT_SEED = 1


def epsilon_at(episode):
    frac = min(1.0, episode / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


def greedy_action(q_table, state):
    return max(range(CartPoleEnv.N_ACTIONS), key=lambda a: q_table[state][a])


def train(env, rng):
    q_table = [[0.0] * env.N_ACTIONS for _ in range(env.n_states)]
    returns = []

    for episode in range(N_EPISODES):
        state = env.reset()
        eps = epsilon_at(episode)
        total_reward = 0.0

        while True:
            if rng.random() < eps:
                action = rng.randrange(env.N_ACTIONS)
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

        if (episode + 1) % LOG_INTERVAL == 0:
            print(f"episode {episode + 1:>6}  "
                  f"avg_return(last {LOG_INTERVAL})={mean(returns[-LOG_INTERVAL:]):.1f}  "
                  f"eps={eps:.3f}")

    return q_table


def evaluate(env, policy, n_episodes):
    """Run a fixed policy; return (mean_return, std_return, solved_rate)."""
    returns = []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        while True:
            state, reward, terminated, truncated, _ = env.step(policy(state))
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
    solved = sum(1 for r in returns if r >= env.MAX_STEPS) / n_episodes
    return mean(returns), pstdev(returns), solved


def main():
    env = CartPoleEnv(num_bins=NUM_BINS, seed=ENV_SEED)
    rng = random.Random(AGENT_SEED)

    q_table = train(env, rng)

    random_policy = evaluate(env, lambda s: rng.randrange(env.N_ACTIONS), N_EVAL_EPISODES)
    trained = evaluate(env, lambda s: greedy_action(q_table, s), N_EVAL_EPISODES)

    print()
    print(f"NUM_BINS={NUM_BINS}  n_states={env.n_states}  MAX_STEPS={env.MAX_STEPS}")
    print(f"{'policy':<20}{'mean_return':>14}{'std_return':>12}{'solved_rate(=MAX_STEPS)':>26}")
    for name, (ret, std_ret, solved) in [
        ("Random", random_policy),
        ("Trained Tabular Q", trained),
    ]:
        print(f"{name:<20}{ret:>14.2f}{std_ret:>12.2f}{solved:>26.2%}")


if __name__ == "__main__":
    main()
