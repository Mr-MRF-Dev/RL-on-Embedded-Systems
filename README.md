# 🏫 RL on Embedded Systems

Implementation and evaluation of Reinforcement Learning algorithms on embedded systems, focusing on performance, resource constraints, and real-time interaction.

## 📌 Overview

Tabular Q-Learning and DQN on micro-controllers (Arduino Uno/Nano, NodeMCU, ESP32, STM32F4), trained **on the board** wherever memory allows. Every problem has a Python reference (`pc/`) and Arduino sketches (`arduino/`) built from the same environment and constants.

## 📁 Structure

```txt
.
├── src/
│   ├── dpm/        # Dynamic Power Management (+ Safety Shield)
│   ├── grid/       # Grid-World navigation (Q-Learning, DQN)
│   └── cartpole/   # Cart-Pole balancing (Q-Learning, DQN)
├── tools/          # weights exporter (PyTorch -> C) and live serial visualizer
└── requirements.txt
```

Each problem folder has its own README.

## 📊 Results

- 🔋 **DPM** – always-active drains the battery in 33 steps; the learned policy survives all 200 and beats the threshold baseline (return 108.4 vs 97.3).
- 🛡️ **Safety Shield** – a single hard-block shield made the agent passive near the threshold; a two-tier shield (hard block + warning zone) fixed it.
- 🧭 **Grid-World** – the Q-table needs 16·N² bytes, so boards hit a memory wall; a 388-parameter DQN reaches the optimal return on a 30×30 grid.
- 🎯 **Cart-Pole** – binning limits both memory and solution quality; DQN learns from the continuous state without it.
- ⚡ **Latency** – STM32F4 inference matches or beats PC PyTorch, so PC/Pi latency reflects framework overhead rather than hardware limits.

## 🚀 Quick start

```bash
pip install -r requirements.txt
python src/dpm/pc/train_qlearning.py
```

Sketches: open `src/<problem>/arduino/<sketch>/<sketch>.ino` in the Arduino IDE (115200 baud).
