"""Tabular Q-Learning on the DPM environment with a rule-based Safety Shield.

The shield sits between the agent and the environment and is active during
both training and evaluation. Q-updates use the action that was actually
executed (after any override).

Variants compared, for several seeds:
  1. WITHOUT shield (baseline)
  2. WITH shield (no penalty)   - hard tier only
  3. WITH shield + penalty      - hard tier + flat penalty when it intervenes
  4. WITH two-tier shield       - hard tier + graded warning penalty
                                  (same design as the Arduino sketch)

Usage:
    python train_qlearning_shield.py
"""

import random
from statistics import mean, pstdev

from env import DPMEnv

# Shield
SHIELD_THRESHOLD = 15.0  # hard tier: force Sleep at or below this battery level
WARNING_THRESHOLD = 30.0  # warning tier: reward penalty below this level
WARN_WEIGHT = 4.0  # max warning penalty (reached at battery = 0)
INTERVENTION_PENALTY = 2.0  # flat penalty per override (variant 3 only)
SLEEP_ACTION = 2

# Hyperparameters (same as train_qlearning.py)
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 2500
N_EPISODES = 6000
N_EVAL_EPISODES = 100

SEEDS = (0, 1, 2)

VARIANTS = [
    ("WITHOUT shield (baseline)", dict(use_shield=False)),
    ("WITH shield (no penalty)", dict(use_shield=True)),
    ("WITH shield + penalty", dict(use_shield=True, intervention_penalty=True)),
    ("WITH two-tier shield", dict(use_shield=True, warning_tier=True)),
]


def shield(battery_level, proposed_action):
    """Return the action actually executed."""
    if battery_level <= SHIELD_THRESHOLD and proposed_action != SLEEP_ACTION:
        return SLEEP_ACTION
    return proposed_action


def warning_penalty(battery_level):
    if battery_level < WARNING_THRESHOLD:
        return WARN_WEIGHT * (WARNING_THRESHOLD - battery_level) / WARNING_THRESHOLD
    return 0.0


def epsilon_at(episode):
    frac = min(1.0, episode / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


def greedy_action(q_table, state):
    return max(range(DPMEnv.N_ACTIONS), key=lambda a: q_table[state][a])


def train(env, rng, use_shield, intervention_penalty=False, warning_tier=False):
    """Return (q_table, depletion_events, shield_interventions)."""
    q_table = [[0.0] * DPMEnv.N_ACTIONS for _ in range(DPMEnv.N_STATES)]
    depletion_events = 0  # training episodes that ended with an empty battery
    shield_interventions = 0  # steps where the shield overrode the proposed action

    for episode in range(N_EPISODES):
        state = env.reset()
        eps = epsilon_at(episode)

        while True:
            if rng.random() < eps:
                proposed = rng.randrange(DPMEnv.N_ACTIONS)
            else:
                proposed = greedy_action(q_table, state)

            intervened = False
            action = proposed
            if use_shield:
                action = shield(env.battery, proposed)
                intervened = action != proposed
                shield_interventions += intervened

            battery_before = env.battery
            next_state, reward, terminated, truncated, _ = env.step(action)

            if intervention_penalty and intervened:
                reward -= INTERVENTION_PENALTY
            if warning_tier:
                reward -= warning_penalty(battery_before)

            target = reward if terminated else reward + GAMMA * max(q_table[next_state])
            q_table[state][action] += ALPHA * (target - q_table[state][action])

            state = next_state
            if terminated:
                depletion_events += 1
            if terminated or truncated:
                break

    return q_table, depletion_events, shield_interventions


def evaluate(env, q_table, use_shield):
    """Greedy evaluation; return (mean_return, std_return, mean_survival, std_survival)."""
    returns, survivals = [], []
    for _ in range(N_EVAL_EPISODES):
        state = env.reset()
        total_reward = 0.0
        while True:
            proposed = greedy_action(q_table, state)
            action = shield(env.battery, proposed) if use_shield else proposed
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
        survivals.append(env.t)
    return mean(returns), pstdev(returns), mean(survivals), pstdev(survivals)


def run(label, seed, **variant):
    env = DPMEnv(seed=seed)
    rng = random.Random(seed + 1000)

    q_table, depletions, interventions = train(env, rng, **variant)
    ret, std_ret, surv, std_surv = evaluate(env, q_table, variant["use_shield"])

    print(f"--- {label} (seed={seed}) ---")
    print(f"training depletion events (out of {N_EPISODES} episodes): {depletions}")
    if variant["use_shield"]:
        print(
            f"shield interventions (steps where shield overrode the proposed action): "
            f"{interventions}"
        )
    print(
        f"eval: mean_return={ret:.2f}  std_return={std_ret:.2f}  "
        f"mean_survival={surv:.2f}  std_survival={std_surv:.2f}"
    )
    print()


def main():
    for seed in SEEDS:
        for label, variant in VARIANTS:
            run(label, seed, **variant)


if __name__ == "__main__":
    main()
