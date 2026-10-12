# Cart-Pole Balancing

Tabular Q-Learning on Cart-Pole with a discretized state space, used to study how the
Q-table grows with the number of bins per dimension (curse of dimensionality), plus a DQN
variant that works on the continuous state. Two implementations of the same problem:

- `pc/` – Python reference: environment + training + comparison with a random policy.
- `arduino/` – the same physics and learning loop running **entirely on a micro-controller**.

See [DQN](#dqn) for the neural-network variant.

## Problem

|                    |                                                                                 |
| ------------------ | ------------------------------------------------------------------------------- |
| **Physics**        | standard CartPole-v1 (Barto, Sutton & Anderson)                                 |
| **State**          | 4-D continuous: cart position, cart velocity, pole angle, pole angular velocity |
| **Discretization** | `NUM_BINS` uniform bins per dimension → `NUM_BINS⁴` states                      |
| **Actions**        | 0 = push left, 1 = push right (force ±10 N)                                     |
| **Reward**         | +1 per step                                                                     |
| **Termination**    | pole beyond ±12°, cart beyond ±2.4 m, or 200 steps                              |

The Q-table holds `NUM_BINS⁴ × 2` floats, i.e. **8 · NUM_BINS⁴ bytes**
(`NUM_BINS = 3` → 648 B, `4` → 2 KB, `6` → 10 KB).

## Structure

```txt
cartpole/
├── README.md
├── pc/
│   ├── env.py                      # Cart-Pole environment
│   ├── train_qlearning.py          # tabular training + evaluation
│   ├── train_dqn.py                # DQN training (PyTorch), saves weights/
│   └── weights/
│       └── cartpole_dqn.pt
└── arduino/
    ├── cartpole_qlearning/         # tabular, on-device training
    ├── cartpole_dqn_inference/     # DQN forward pass only (+ weights header)
    ├── cartpole_dqn_demo/          # live stream for tools/rl_serial_visualizer.py
    └── cartpole_dqn_train/         # DQN trained on the microcontroller
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

|               | Episodes | Epsilon decay | Log interval |
| ------------- | -------- | ------------- | ------------ |
| `0` (default) | 10000    | 4000          | 500          |
| `1`           | 50000    | 20000         | 2500         |

The extended budget is for checking whether a drop in solved rate at larger `NUM_BINS`
comes from the fixed episode budget rather than from memory.

## DQN

DQN works on the raw continuous state, so no discretization is needed: the network
(4 → 24 → 24 → 2, 770 parameters, 3080 bytes) is tiny and its size is independent of any bin count.

**1. Train on the PC** (PyTorch; `pip install torch numpy`):

```bash
cd pc
python train_dqn.py
```

The best checkpoint seen during training (short greedy probe every 25 episodes) is evaluated
and saved to `weights/cartpole_dqn.pt`. The unmodified script can also be run on a Raspberry Pi
as a control group.

**2. Export to C**:

```bash
python tools/export_weights_to_c.py src/cartpole/pc/weights/cartpole_dqn.pt cartpole_dqn_weights.h cartpole
```

Copy the header next to `cartpole_dqn_inference.ino` and `cartpole_dqn_demo.ino` (a copy for the
included weights is already there).

**3. Run on the micro-controller** (115200 baud):

| Sketch                   | Purpose                                                                                                                                                                 | Boards                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `cartpole_dqn_inference` | greedy evaluation of the PC-trained network; prints mean return, solved rate, inference latency                                                                         | all (weights live in flash)                                                                                |
| `cartpole_dqn_demo`      | streams every step for `tools/rl_serial_visualizer.py`                                                                                                                  | all                                                                                                        |
| `cartpole_dqn_train`     | full DQN training (replay buffer, target network, Adam, backprop) on the board; prints memory, timing, numerical-stability report and the trained weights as a C header | STM32F4 (tested on STM32F407-Discovery; needs ~80 KB RAM, Uno and ESP8266 stop at compile time on purpose) |

Its replay buffer holds 1500 transitions (PC: 20000) because the buffer is the largest RAM consumer; the measured ceiling on the STM32F407 is 2500.

**Finding:** the last-episode weights are not necessarily the best ones. In the first PC run the
training log (with 5% random actions) reached the maximum score repeatedly, but the final greedy
evaluation only reached about 122, so the best checkpoint is selected instead.

## Reference result (tabular, PC, 6 bins, fixed seeds)

| Policy            | Mean return | Solved rate (200 steps) |
| ----------------- | ----------- | ----------------------- |
| Random            | 23.44       | 0.00%                   |
| Trained Tabular Q | 193.33      | 81.00%                  |

Cart-Pole results vary noticeably between runs (seeds); treat single runs with care.
