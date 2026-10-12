# 📄 Project Report

## 🎯 Goal

How far can reinforcement learning go on embedded hardware? Measured per board: decision quality, latency and memory, using an algorithm ladder of **threshold baseline → Tabular Q-Learning → DQN**.

## 🧪 Setup

- **Boards:** Arduino Uno (AVR, 2 KB RAM), NodeMCU ESP8266, STM32F407-Discovery; Raspberry Pi 4 and a PC as control groups.
- **Problems:** Dynamic Power Management, Grid-World, Cart-Pole, plus a rule-based Safety Shield on top of DPM.
- **DQN protocol:** PC training → Raspberry Pi (same code) → MCU inference → full training on the MCU.

## 📊 What happened

| Problem                | Outcome                                                                                                                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🔋 DPM                 | On-device Q-Learning on Uno survives all 200 steps and beats the threshold baseline (return 116.6 vs 97.3). NodeMCU is 10× faster per step but learned a worse policy (single run, 172 steps).                                |
| 🛡️ Safety Shield       | A hard-block shield alone cut survival from 200 to ~125 steps (over-reliance on the shield). Adding a graded warning zone restored it in Python and on both boards (200/200 on hardware), though Python results vary by seed. |
| 🧭 Grid-World          | Uno's memory wall follows `16·N² + 252` bytes exactly. NodeMCU failed at N=30 and 45 long before running out of memory; 15× more training episodes fixed it.                                                                  |
| 🎯 Cart-Pole           | Uno only fits 3 bins, which never solves it (0%); 4 bins don't fit. On NodeMCU, quality drops with more bins at a fixed budget and varies a lot between runs.                                                                 |
| 🧠 DQN (PC, Pi)        | Raspberry Pi 4 reproduces PC results (Cart-Pole ~197.6, Grid N=30 optimal at −7.0), training 7.5–8.7× slower.                                                                                                                 |
| ⚙️ DQN inference       | Same weights give the same quality on Uno, NodeMCU and STM32F407. Latency: STM32F407 ≈ 36–68 µs (on par with PyTorch on a PC), NodeMCU ≈ 0.6–1.3 ms, Uno ≈ 5.6–12 ms.                                                         |
| 🔁 DQN training on MCU | STM32F407 learned optimal policies from scratch: Cart-Pole in 12 min, Grid N=10 in 7 min, Grid N=30 in 5.6 h (5–8× slower than PC).                                                                                           |

## 🔑 Key findings

- **Hard wall vs. soft wall:** memory failures are exact and predictable (compile/link error); a separate convergence wall appears when the training budget is fixed while the problem grows, and it disappears with more episodes.
- **Best-checkpoint selection is essential for DQN:** last-episode weights were often much worse (Cart-Pole 0% → 85% solved on PC; 0% vs 100% on the STM32F407).
- **DQN weights have constant size** (1.5–3 KB) while Q-tables grow with the problem, but the advantage depends on enough training budget.
- **Replay buffer is the limit for on-device DQN training**, not the network (measured ceilings on STM32F407: 2500 for Cart-Pole, 4750 for Grid-World).
- **Results vary between seeds**, especially on Cart-Pole; single runs should not be over-interpreted.

## 🏁 Conclusion

Autonomous RL, from tabular methods to DQN with back-propagation, runs on small micro-controllers. The real constraint is usually the time and resources needed to learn, so the choice is between learning on the device and deploying weights trained elsewhere.
