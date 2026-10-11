# Cart-Pole Balancing

Tabular Q-Learning on Cart-Pole with a discretized state space, used to study how the
Q-table grows with the number of bins per dimension (curse of dimensionality).
Two implementations of the same problem:

- `pc/` – Python reference: environment + training + comparison with a random policy.
- `arduino/` – the same physics and learning loop running **entirely on a micro-controller**.

## Problem

| | |
| --- | --- |
| **Physics** | standard CartPole-v1 (Barto, Sutton & Anderson) |
| **State** | 4-D continuous: cart position, cart velocity, pole angle, pole angular velocity |
| **Discretization** | `NUM_BINS` uniform bins per dimension → `NUM_BINS⁴` states |
| **Actions** | 0 = push left, 1 = push right (force ±10 N) |
| **Reward** | +1 per step |
| **Termination** | pole beyond ±12°, cart beyond ±2.4 m, or 200 steps |

The Q-table holds `NUM_BINS⁴ × 2` floats, i.e. **8 · NUM_BINS⁴ bytes**
(`NUM_BINS = 3` → 648 B, `4` → 2 KB, `6` → 10 KB).

## Structure

```txt
cartpole/
├── README.md
├── pc/
│   ├── env.py                      # Cart-Pole environment
│   └── train_qlearning.py          # training + evaluation
└── arduino/
    └── cartpole_qlearning/
        └── cartpole_qlearning.ino  # on-device training + evaluation
```

The sketch lives in a folder with the same name as the `.ino` file (required by the Arduino IDE).
Constants in `env.py`, `train_qlearning.py` and `cartpole_qlearning.ino` are kept identical.

## Run

**PC** (Python 3, no dependencies):

```bash
cd pc
python train_qlearning.py
```

**MicroController:**

1. Open `arduino/cartpole_qlearning/cartpole_qlearning.ino` in the Arduino IDE.
2. Set `NUM_BINS` (default 6; an Arduino Uno only fits very small values such as 3).
3. Select the board (Uno/Nano, ESP8266, ESP32, STM32 via STM32duino) and upload.
4. Open the Serial Monitor at **115200 baud**.

Serial output (CSV): `NUM_BINS`, `N_STATES`, Q-table size in bytes, a training log
(`episode,return,avg_step_us,epsilon`), and a greedy evaluation summary
(mean/std return and solved rate over 100 episodes; solved = reached 200 steps).

## Memory scaling study

Change `NUM_BINS` and recompile (uploading is not needed for this). The compiler reports
RAM usage; the size at which linking fails with a RAM overflow is the breaking point for
that board. Memory limits both the problem size and the solution quality: with very few
bins the Q-table fits, but the discretization is too coarse to balance the pole reliably.

## Training budget

`#define EXTENDED_BUDGET` selects the training budget:

| | Episodes | Epsilon decay | Log interval |
| --- | --- | --- | --- |
| `0` (default) | 10000 | 4000 | 500 |
| `1` | 50000 | 20000 | 2500 |

The extended budget is for checking whether a drop in solved rate at larger `NUM_BINS`
comes from the fixed episode budget rather than from memory.

## Reference result (PC, 6 bins, fixed seeds)

| Policy | Mean return | Solved rate (200 steps) |
| --- | --- | --- |
| Random | 23.44 | 0.00% |
| Trained Tabular Q | 193.33 | 81.00% |

Cart-Pole results vary noticeably between runs (seeds); treat single runs with care.
