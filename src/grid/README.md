# Grid-World Navigation

Tabular Q-Learning on an N × N grid, used to study how the Q-table's memory
requirement scales with problem size, plus a DQN variant whose size does not depend on N.
Two implementations of the same problem:

- `pc/` – Python reference: environment + training + comparison with a random policy.
- `arduino/` – the same environment and learning loop running **entirely on a micro-controller**.

See [DQN](#dqn) for the neural-network variant.

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
│   ├── env.py                      # Grid-World environment
│   ├── train_qlearning.py          # tabular training + evaluation
│   ├── train_dqn.py                # DQN training (PyTorch), saves weights/
│   └── weights/
│       ├── grid_dqn_n10.pt
│       └── grid_dqn_n30.pt
└── arduino/
    ├── grid_qlearning/             # tabular, on-device training
    ├── grid_dqn_inference/         # DQN forward pass only (+ weights header)
    ├── grid_dqn_demo/              # live stream for tools/rl_serial_visualizer.py
    └── grid_dqn_train/             # DQN trained on the micro-controller
```

Each sketch lives in a folder with the same name as its `.ino` file (required by the Arduino IDE).
Constants in the Python files and the sketches are kept identical.

## Run (tabular Q-Learning)

**PC** (Python 3, no dependencies):

```bash
cd pc
python train_qlearning.py
```

**MicroController:**

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

## DQN

The observation is the normalized `(row, col)` position, so the network (2 → 16 → 16 → 4,
388 parameters, 1552 bytes) has the same size for every `GRID_N`, unlike the Q-table.

**1. Train on the PC** (PyTorch; `pip install torch numpy`):

```bash
cd pc
python train_dqn.py
```

Set `GRID_PRESET` at the top of the script: `1` = `GRID_N` 10, 3000 episodes (quick check);
`2` = `GRID_N` 30, 30000 episodes. The best checkpoint seen during training (short greedy probe
every `PROBE_INTERVAL` episodes) is evaluated and saved to `weights/grid_dqn_n<GRID_N>.pt`.
The unmodified script can also be run on a Raspberry Pi as a control group.

**2. Export to C**:

```bash
python tools/export_weights_to_c.py src/grid/pc/weights/grid_dqn_n30.pt grid_dqn_weights.h grid
```

Copy the header next to `grid_dqn_inference.ino` and `grid_dqn_demo.ino` (a copy for `GRID_N` 30
is already included).

**3. Run on the micro-controller** (115200 baud; set `GRID_N` to match the weights):

| Sketch | Purpose | Boards |
| --- | --- | --- |
| `grid_dqn_inference` | greedy evaluation of the PC-trained network; prints mean return, steps, goal-reach rate, inference latency | all (weights live in flash) |
| `grid_dqn_demo` | streams every step for `tools/rl_serial_visualizer.py` | all |
| `grid_dqn_train` | full DQN training (replay buffer, target network, Adam, backprop) on the board; prints memory, timing, numerical-stability report and the trained weights as a C header | ESP32, STM32F4 (needs ~130 KB RAM; Uno and ESP8266 stop at compile time on purpose) |

`grid_dqn_train` has the same `GRID_PRESET` switch as the PC script. Its replay buffer holds
5000 transitions (PC: 20000) because the buffer is the largest RAM consumer.

### Findings

- With the fixed 3000-episode budget DQN solves 10 × 10 and 15 × 15 optimally but fails from
  20 × 20 on (0% goal-reach). Likely cause: random exploration almost never reaches the goal, so
  the replay buffer holds no positive reward. With 30000 episodes plus checkpoint selection, 30 × 30
  reaches the optimal return (−7.0, 100% goal-reach).
- The last-episode weights are not necessarily the best ones, hence the checkpoint selection.

## Reference result (tabular, PC, 6 × 6, fixed seeds)

| Policy | Mean return | Mean steps | Goal-reach rate |
| --- | --- | --- | --- |
| Random | -52.83 | 45.6 | 21.00% |
| Trained Tabular Q | 41.00 | 10.0 | 100.00% |

The optimal return for 6 × 6 is 41.0.
