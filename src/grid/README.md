# Grid-World Navigation

Tabular Q-Learning on an N × N grid, used to study how the Q-table's memory
requirement scales with problem size. Two implementations of the same problem:

- `pc/` – Python reference: environment + training + comparison with a random policy.
- `arduino/` – the same environment and learning loop running **entirely on a microcontroller**.

## Problem

| | |
| --- | --- |
| **Grid** | N × N, start at (0, 0), goal at (N-1, N-1), no obstacles |
| **State** | `row * N + col` (N² states) |
| **Actions** | 0 = Up, 1 = Down, 2 = Left, 3 = Right |
| **Reward** | −1 per step, extra −2 for hitting a wall (agent stays in place), +50 at the goal |
| **Episode** | up to `max(50, 6N)` steps |

The Q-table holds N² × 4 floats, i.e. **16 · N² bytes**.

## Structure

```txt
grid/
├── README.md
├── pc/
│   ├── env.py                  # Grid-World environment
│   └── train_qlearning.py      # training + evaluation
└── arduino/
    └── grid_qlearning/
        └── grid_qlearning.ino  # on-device training + evaluation
```

The sketch lives in a folder with the same name as the `.ino` file (required by the Arduino IDE).
Constants in `env.py`, `train_qlearning.py` and `grid_qlearning.ino` are kept identical.

## Run

**PC** (Python 3, no dependencies):

```bash
cd pc
python train_qlearning.py
```

**Microcontroller:**

1. Open `arduino/grid_qlearning/grid_qlearning.ino` in the Arduino IDE.
2. Set `GRID_N` (default 6).
3. Select the board (Uno/Nano, ESP8266, ESP32, STM32 via STM32duino) and upload.
4. Open the Serial Monitor at **115200 baud**.

Serial output (CSV): `GRID_N`, `N_STATES`, Q-table size in bytes, a training log
(`episode,return,steps,avg_step_us,epsilon`), and a greedy evaluation summary
(mean/std return, mean/std steps, goal-reach rate over 100 episodes).

## Memory scaling study

To find the largest grid a board can hold, change `GRID_N` and recompile (uploading is
not needed for this). The compiler reports RAM usage; the size at which linking fails
with a RAM overflow is the breaking point for that board. Only sizes that fit need to
be uploaded to collect timing and return metrics.

## Training budget

`#define EXTENDED_BUDGET` selects the training budget:

| | Episodes | Epsilon decay | Log interval |
| --- | --- | --- | --- |
| `0` (default) | 2000 | 800 | 50 |
| `1` | 30000 | 12000 | 1000 |

The extended budget is for checking whether a failure on a large grid comes from the
fixed episode budget rather than from memory.

## Reference result (PC, 6 × 6, fixed seeds)

| Policy | Mean return | Mean steps | Goal-reach rate |
| --- | --- | --- | --- |
| Random | -52.83 | 45.6 | 21.00% |
| Trained Tabular Q | 41.00 | 10.0 | 100.00% |

The optimal return for 6 × 6 is 41.0.
