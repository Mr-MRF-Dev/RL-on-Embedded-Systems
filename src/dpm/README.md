# Dynamic Power Management (DPM)

Tabular Q-Learning for a battery-powered device that chooses a power mode at every step.
Two implementations of the same problem:

- `pc/` – Python reference: environment + training + baseline comparison.
- `arduino/` – the same environment and learning loop running **entirely on a micro-controller**.

A rule-based **Safety Shield** variant is included for both (see [Safety Shield](#safety-shield)).

## Problem

|              |                                                                                                |
| ------------ | ---------------------------------------------------------------------------------------------- |
| **State**    | 20 discrete states = 5 battery bins × 4 idle-time bins                                         |
| **Actions**  | 0 = Active, 1 = Idle, 2 = Sleep                                                                |
| **Workload** | a request arrives each step with probability 0.30                                              |
| **Reward**   | `+1.5` survival bonus − `power_cost / 3` − latency penalty, `−50` when the battery is depleted |
| **Episode**  | up to 200 steps, battery starts at 100                                                         |

Latency penalty if a request arrives: Active 0, Idle 1, Sleep 3.
The battery is simulated; no real energy is measured.

## Structure

```txt
dpm/
├── README.md
├── pc/
│   ├── env.py                          # DPM environment
│   ├── train_qlearning.py              # training + evaluation vs. baselines
│   └── train_qlearning_shield.py       # shielded vs. unshielded comparison
└── arduino/
    ├── dpm_qlearning/
    │   └── dpm_qlearning.ino           # on-device training + evaluation
    └── dpm_qlearning_shield/
        └── dpm_qlearning_shield.ino    # same, with the Safety Shield
```

Each sketch lives in a folder with the same name as its `.ino` file (required by the Arduino IDE).
Constants in the Python files and the sketches are kept identical.

## Run

**PC** (Python 3, no dependencies):

```bash
cd pc
python train_qlearning.py
```

Trains for 6000 episodes, then evaluates Always-Active, a Threshold baseline, and the trained agent.

**Microcontroller:**

1. Open `arduino/dpm_qlearning/dpm_qlearning.ino` in the Arduino IDE.
2. Select the board (Uno/Nano, ESP8266, ESP32, STM32 via STM32duino) and upload.
3. Open the Serial Monitor at **115200 baud**.

Serial output (CSV): Q-table size in bytes, a training log every 50 episodes
(`episode,return,survival_steps,avg_step_us,epsilon`), a greedy evaluation summary
(mean/std over 100 episodes), and the learned policy per state.

## Safety Shield

A small deterministic layer between the agent and the environment, active during both
training and evaluation. Q-updates use the action that was actually executed.

| Tier    | Condition    | Effect                                                |
| ------- | ------------ | ----------------------------------------------------- |
| Hard    | battery ≤ 15 | proposed action is overridden with Sleep              |
| Warning | battery < 30 | no override; reward penalty `4 · (30 − battery) / 30` |

A hard tier alone made results worse: the agent learns to spend battery early and rely on
being forced into Sleep, which in the Python experiments cut mean survival to about 124–128
of 200 steps (seeds 0–2) versus 150–200 without a shield. The warning tier adds a learning
signal before the cutoff. Adding a flat penalty only at the moment of override did not help.

Run the comparison (4 variants × 3 seeds):

```bash
cd pc
python train_qlearning_shield.py
```

The Arduino sketch `dpm_qlearning_shield` uses the two-tier design and additionally logs
the number of shield interventions and depletion episodes during training.

## Reference result (PC, fixed seeds)

| Policy             | Avg return | Avg survival (steps) |
| ------------------ | ---------- | -------------------- |
| Always-Active      | -34.60     | 33.0                 |
| Threshold baseline | 97.27      | 200.0                |
| Trained Tabular Q  | 108.43     | 200.0                |
