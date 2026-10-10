# Dynamic Power Management (DPM)

Tabular Q-Learning for a battery-powered device that chooses a power mode at every step.
Two implementations of the same problem:

- `pc/` – Python reference: environment + training + baseline comparison.
- `arduino/` – the same environment and learning loop running **entirely on a microcontroller**.

## Problem

| | |
| --- | --- |
| **State** | 20 discrete states = 5 battery bins × 4 idle-time bins |
| **Actions** | 0 = Active, 1 = Idle, 2 = Sleep |
| **Workload** | a request arrives each step with probability 0.30 |
| **Reward** | `+1.5` survival bonus − `power_cost / 3` − latency penalty, `−50` when the battery is depleted |
| **Episode** | up to 200 steps, battery starts at 100 |

Latency penalty if a request arrives: Active 0, Idle 1, Sleep 3.
The battery is simulated; no real energy is measured.

## Structure

```txt
dpm/
├── README.md
├── pc/
│   ├── env.py                  # DPM environment
│   └── train_qlearning.py      # training + evaluation vs. baselines
└── arduino/
    └── dpm_qlearning/
        └── dpm_qlearning.ino   # on-device training + evaluation
```

The sketch lives in a folder with the same name as the `.ino` file (required by the Arduino IDE).
Constants in `env.py`, `train_qlearning.py` and `dpm_qlearning.ino` are kept identical.

## Run

**PC** (Python 3, no dependencies):

```bash
cd pc
python train_qlearning.py
```

Trains for 6000 episodes, then evaluates Always-Active, a Threshold baseline, and the trained agent.

**MicroController:**

1. Open `arduino/dpm_qlearning/dpm_qlearning.ino` in the Arduino IDE.
2. Select the board (Uno/Nano, ESP8266, ESP32, STM32 via STM32duino) and upload.
3. Open the Serial Monitor at **115200 baud**.

Serial output (CSV): Q-table size in bytes, a training log every 50 episodes
(`episode,return,survival_steps,avg_step_us,epsilon`), a greedy evaluation summary
(mean/std over 100 episodes), and the learned policy per state.

## Reference result (PC, fixed seeds)

| Policy | Avg return | Avg survival (steps) |
| --- | --- | --- |
| Always-Active | -34.60 | 33.0 |
| Threshold baseline | 97.27 | 200.0 |
| Trained Tabular Q | 108.43 | 200.0 |
