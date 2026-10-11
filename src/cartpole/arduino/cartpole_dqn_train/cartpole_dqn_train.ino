/*
  Cart-Pole on-device DQN training (backpropagation on the microcontroller)

  Mirrors pc/train_dqn.py: 4 -> 24 -> 24 -> 2 network, replay buffer, target
  network, Adam, epsilon-greedy and best-checkpoint selection, all in float32
  and plain Arduino-core C++. One sample at a time is forwarded and
  back-propagated and the gradients are accumulated over the mini-batch, which
  gives the same result as a batched pass for a mean-reduced MSE loss without
  storing per-batch activations.

  Boards: ESP32 and STM32F4 (STM32duino). The sketch needs ~130 KB of static RAM at
  the default REPLAY_CAPACITY, so Arduino Uno (2 KB) and ESP8266 stop at compile time on purpose; define
  FORCE_SMALL_BOARD_ATTEMPT to bypass the guard and capture the linker error.
  Serial: 115200 baud.

  Deviations from the PC script
    - REPLAY_CAPACITY is 2750 here (20000 on the PC): the replay buffer is the
      largest RAM consumer (40 bytes per transition). Raise it to find the
      board's ceiling; the sketch prints free RAM before and after allocation.
    - No PyTorch: nn.Linear's default initialisation and Adam's update rule
      are re-implemented by hand.

  Output (CSV and plain text)
    - board, free RAM, static memory breakdown, configuration
    - every LOG_INTERVAL episodes: episode,avg_return,eps,avg_loss,avg_train_step_us,free_ram_b
    - checkpoint-selection events (greedy probe score)
    - training wall-clock time, greedy forward-pass and full training-step latency
    - float32 numerical-stability report (max |weight|, NaN/Inf check)
    - final greedy evaluation of the best and of the last-episode weights
    - optionally the trained weights as a C header (PRINT_FINAL_WEIGHTS), a
      drop-in replacement for cartpole_dqn_weights.h in cartpole_dqn_inference
*/

#include <math.h>
#include <stdint.h>
#include <string.h>

#if (defined(__AVR__) || defined(ESP8266)) && !defined(FORCE_SMALL_BOARD_ATTEMPT)
  #error "On-device DQN training needs ~130 KB of SRAM; Arduino Uno and ESP8266 cannot host it. Use cartpole_dqn_inference.ino on these boards."
#endif

// ---------------------- Board helpers ----------------------
#if defined(ESP32) || defined(ESP8266)
  int32_t freeRamBytes() { return (int32_t)ESP.getFreeHeap(); }
#elif defined(__arm__)
  extern "C" char *sbrk(int incr);
  int32_t freeRamBytes() {
    char stackTop;
    return (int32_t)(&stackTop - (char *)sbrk(0));
  }
#else
  extern int __heap_start, *__brkval;
  int32_t freeRamBytes() {
    int v;
    return (int32_t)((int)&v - (__brkval == 0 ? (int)&__heap_start : (int)__brkval));
  }
#endif

// NaN / Inf test written by hand: isnan()/isinf() availability varies by core
static inline bool isBadFloat(float v) {
  if (!(v == v)) return true;                  // NaN
  return (v > 3.0e38f) || (v < -3.0e38f);      // +/-Inf
}

static const char *boardName() {
#if defined(STM32F4) || defined(ARDUINO_DISCO_F407VG) || defined(ARDUINO_ARCH_STM32)
  return "STM32F4 (Cortex-M4F, hardware FPU)";
#elif defined(ESP32)
  return "ESP32";
#elif defined(ESP8266)
  return "ESP8266 / NodeMCU";
#elif defined(__AVR__)
  return "AVR (Arduino Uno/Nano)";
#else
  return "unknown";
#endif
}

// ---------------------- Configuration (matches pc/train_dqn.py unless noted) ----------------------
const uint32_t SEED = 12345u;   // fixed, so on-device runs are reproducible

const uint16_t N_EPISODES      = 800;
const uint16_t N_EVAL_EPISODES = 100;
const uint16_t LOG_INTERVAL    = 25;

// Checkpoint selection: the last weights are not necessarily the best ones
const uint16_t PROBE_INTERVAL = 25;
const uint8_t  PROBE_EPISODES = 10;

const float    GAMMA      = 0.99f;
const float    LR         = 1e-3f;
const uint16_t BATCH_SIZE = 64;

const uint16_t REPLAY_CAPACITY           = 2750;   // PC: 20000
const uint16_t MIN_REPLAY_BEFORE_TRAIN   = 1000;
const uint32_t TARGET_UPDATE_EVERY_STEPS = 500;

const float    EPS_START          = 1.0f;
const float    EPS_END            = 0.05f;
const uint16_t EPS_DECAY_EPISODES = 400;

// Adam, same defaults as torch.optim.Adam
const float ADAM_BETA1 = 0.9f;
const float ADAM_BETA2 = 0.999f;
const float ADAM_EPS   = 1e-8f;

// Per-element gradient clipping; 0 = disabled (as on the PC)
const float GRAD_CLIP_ABS = 0.0f;

// Print the board-trained weights as a C header at the end (1 = yes)
#define PRINT_FINAL_WEIGHTS 1

// ---------------------- Network: 4 -> 24 -> 24 -> 2 ----------------------
const int OBS_DIM   = 4;
const int N_ACTIONS = 2;
const int H1 = 24;
const int H2 = 24;

// All parameters live in one flat array (PyTorch layout: W is [out, in],
// row-major), so target copy, gradient and Adam moments are plain loops.
const int OFF_W0 = 0;
const int OFF_B0 = OFF_W0 + H1 * OBS_DIM;
const int OFF_W1 = OFF_B0 + H1;
const int OFF_B1 = OFF_W1 + H2 * H1;
const int OFF_W2 = OFF_B1 + H2;
const int OFF_B2 = OFF_W2 + N_ACTIONS * H2;
const int N_PARAMS = OFF_B2 + N_ACTIONS;    // 770

float gParams[N_PARAMS];   // policy network (being trained)
float gTarget[N_PARAMS];   // target network (periodic hard copy)
float gBest[N_PARAMS];     // best checkpoint seen during training
float gGrad[N_PARAMS];     // gradient accumulator for one mini-batch
float gAdamM[N_PARAMS];    // Adam first moment
float gAdamV[N_PARAMS];    // Adam second moment
uint32_t gAdamT = 0;       // Adam timestep

// Activations of a single sample
struct Activations {
  float in[OBS_DIM];
  float z1[H1], a1[H1];
  float z2[H2], a2[H2];
  float q[N_ACTIONS];
};
Activations gAct;

// ---------------------- Replay buffer ----------------------
struct Transition {
  float   s[OBS_DIM];
  float   s2[OBS_DIM];
  float   r;
  uint8_t a;
  uint8_t done;    // real terminal only: truncation still bootstraps, as on the PC
};

Transition gReplay[REPLAY_CAPACITY];
uint16_t gReplaySize = 0;
uint16_t gReplayHead = 0;   // circular write position

void replayPush(const float *s, uint8_t a, float r, const float *s2, bool terminated) {
  Transition &t = gReplay[gReplayHead];
  memcpy(t.s, s, sizeof(t.s));
  memcpy(t.s2, s2, sizeof(t.s2));
  t.r = r;
  t.a = a;
  t.done = terminated ? 1 : 0;
  gReplayHead = (uint16_t)((gReplayHead + 1) % REPLAY_CAPACITY);
  if (gReplaySize < REPLAY_CAPACITY) gReplaySize++;
}

// ---------------------- PRNG (xorshift32) ----------------------
uint32_t rngState;

uint32_t xorshift32() {
  rngState ^= rngState << 13;
  rngState ^= rngState >> 17;
  rngState ^= rngState << 5;
  return rngState;
}

float randUnit()   { return (xorshift32() >> 8) / 16777216.0f; }  // [0, 1)
float randSmall()  { return (randUnit() - 0.5f) * 0.1f; }         // [-0.05, 0.05)

// ---------------------- Environment (matches pc/env.py) ----------------------
const float GRAVITY         = 9.8f;
const float MASSCART        = 1.0f;
const float MASSPOLE        = 0.1f;
const float TOTAL_MASS      = MASSCART + MASSPOLE;
const float LENGTH          = 0.5f;   // half the pole length
const float POLEMASS_LENGTH = MASSPOLE * LENGTH;
const float FORCE_MAG       = 10.0f;
const float TAU             = 0.02f;  // seconds per step

const float THETA_THRESHOLD = 12.0f * 2.0f * 3.14159265f / 360.0f;
const float X_THRESHOLD     = 2.4f;
const uint16_t MAX_STEPS    = 200;

float cartX, cartXDot, poleTheta, poleThetaDot;
uint16_t stepCount;

void envReset() {
  cartX        = randSmall();
  cartXDot     = randSmall();
  poleTheta    = randSmall();
  poleThetaDot = randSmall();
  stepCount    = 0;
}

void envState(float *out4) {
  out4[0] = cartX;
  out4[1] = cartXDot;
  out4[2] = poleTheta;
  out4[3] = poleThetaDot;
}

float envStep(uint8_t action, bool &terminated, bool &done) {
  stepCount++;

  float force    = (action == 1) ? FORCE_MAG : -FORCE_MAG;
  float costheta = cosf(poleTheta);
  float sintheta = sinf(poleTheta);

  float temp = (force + POLEMASS_LENGTH * poleThetaDot * poleThetaDot * sintheta) / TOTAL_MASS;
  float thetaAcc = (GRAVITY * sintheta - costheta * temp) /
                   (LENGTH * (4.0f / 3.0f - MASSPOLE * costheta * costheta / TOTAL_MASS));
  float xAcc = temp - POLEMASS_LENGTH * thetaAcc * costheta / TOTAL_MASS;

  cartX        += TAU * cartXDot;
  cartXDot     += TAU * xAcc;
  poleTheta    += TAU * poleThetaDot;
  poleThetaDot += TAU * thetaAcc;

  terminated = (cartX < -X_THRESHOLD || cartX > X_THRESHOLD ||
                poleTheta < -THETA_THRESHOLD || poleTheta > THETA_THRESHOLD);
  bool truncated = (stepCount >= MAX_STEPS);
  done = terminated || truncated;
  return 1.0f;
}

// ---------------------- Forward pass ----------------------
void forward(const float *p, const float *in, Activations &act) {
  memcpy(act.in, in, sizeof(act.in));

  for (int o = 0; o < H1; o++) {
    float sum = p[OFF_B0 + o];
    const float *w = p + OFF_W0 + o * OBS_DIM;
    for (int i = 0; i < OBS_DIM; i++) sum += w[i] * act.in[i];
    act.z1[o] = sum;
    act.a1[o] = sum > 0.0f ? sum : 0.0f;
  }
  for (int o = 0; o < H2; o++) {
    float sum = p[OFF_B1 + o];
    const float *w = p + OFF_W1 + o * H1;
    for (int i = 0; i < H1; i++) sum += w[i] * act.a1[i];
    act.z2[o] = sum;
    act.a2[o] = sum > 0.0f ? sum : 0.0f;
  }
  for (int o = 0; o < N_ACTIONS; o++) {
    float sum = p[OFF_B2 + o];
    const float *w = p + OFF_W2 + o * H2;
    for (int i = 0; i < H2; i++) sum += w[i] * act.a2[i];
    act.q[o] = sum;   // raw Q-values, no activation
  }
}

uint8_t argmaxQ(const float *q) {
  uint8_t best = 0;
  for (uint8_t a = 1; a < (uint8_t)N_ACTIONS; a++) if (q[a] > q[best]) best = a;
  return best;
}

float maxQ(const float *q) {
  float m = q[0];
  for (int a = 1; a < N_ACTIONS; a++) if (q[a] > m) m = q[a];
  return m;
}

// ---------------------- Backward pass ----------------------
// Accumulates dLoss/dParam into gGrad for one sample. dq holds dLoss/dq per
// output unit (only the taken action is non-zero for the DQN loss).
void backward(const Activations &act, const float *dq) {
  float dA2[H2];
  for (int i = 0; i < H2; i++) dA2[i] = 0.0f;

  // output layer
  for (int o = 0; o < N_ACTIONS; o++) {
    float g = dq[o];
    if (g == 0.0f) continue;
    gGrad[OFF_B2 + o] += g;
    float *gw = gGrad + OFF_W2 + o * H2;
    const float *w = gParams + OFF_W2 + o * H2;
    for (int i = 0; i < H2; i++) {
      gw[i] += g * act.a2[i];
      dA2[i] += g * w[i];
    }
  }

  // second hidden layer (through ReLU)
  float dA1[H1];
  for (int i = 0; i < H1; i++) dA1[i] = 0.0f;
  for (int o = 0; o < H2; o++) {
    float g = (act.z2[o] > 0.0f) ? dA2[o] : 0.0f;
    if (g == 0.0f) continue;
    gGrad[OFF_B1 + o] += g;
    float *gw = gGrad + OFF_W1 + o * H1;
    const float *w = gParams + OFF_W1 + o * H1;
    for (int i = 0; i < H1; i++) {
      gw[i] += g * act.a1[i];
      dA1[i] += g * w[i];
    }
  }

  // first hidden layer (through ReLU); nothing to propagate into the input
  for (int o = 0; o < H1; o++) {
    float g = (act.z1[o] > 0.0f) ? dA1[o] : 0.0f;
    if (g == 0.0f) continue;
    gGrad[OFF_B0 + o] += g;
    float *gw = gGrad + OFF_W0 + o * OBS_DIM;
    for (int i = 0; i < OBS_DIM; i++) gw[i] += g * act.in[i];
  }
}

void zeroGrad() {
  for (int i = 0; i < N_PARAMS; i++) gGrad[i] = 0.0f;
}

void adamStep() {
  gAdamT++;
  float bc1 = 1.0f - powf(ADAM_BETA1, (float)gAdamT);
  float bc2 = 1.0f - powf(ADAM_BETA2, (float)gAdamT);
  for (int i = 0; i < N_PARAMS; i++) {
    float g = gGrad[i];
    if (GRAD_CLIP_ABS > 0.0f) {
      if (g > GRAD_CLIP_ABS) g = GRAD_CLIP_ABS;
      else if (g < -GRAD_CLIP_ABS) g = -GRAD_CLIP_ABS;
    }
    gAdamM[i] = ADAM_BETA1 * gAdamM[i] + (1.0f - ADAM_BETA1) * g;
    gAdamV[i] = ADAM_BETA2 * gAdamV[i] + (1.0f - ADAM_BETA2) * g * g;
    float mHat = gAdamM[i] / bc1;
    float vHat = gAdamV[i] / bc2;
    gParams[i] -= LR * mHat / (sqrtf(vHat) + ADAM_EPS);
  }
}

// ---------------------- Initialisation ----------------------
// PyTorch's default nn.Linear init: weights and biases ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))
void initLayer(int offW, int offB, int inDim, int outDim) {
  float bound = 1.0f / sqrtf((float)inDim);
  for (int k = 0; k < outDim * inDim; k++)
    gParams[offW + k] = (randUnit() * 2.0f - 1.0f) * bound;
  for (int o = 0; o < outDim; o++)
    gParams[offB + o] = (randUnit() * 2.0f - 1.0f) * bound;
}

void initNetwork() {
  initLayer(OFF_W0, OFF_B0, OBS_DIM, H1);
  initLayer(OFF_W1, OFF_B1, H1, H2);
  initLayer(OFF_W2, OFF_B2, H2, N_ACTIONS);
  for (int i = 0; i < N_PARAMS; i++) { gAdamM[i] = 0.0f; gAdamV[i] = 0.0f; }
  memcpy(gTarget, gParams, sizeof(gParams));
  memcpy(gBest, gParams, sizeof(gParams));
}

// ---------------------- One training step ----------------------
// Samples a mini-batch, computes TD targets with the target network,
// back-propagates the mean-reduced MSE loss and applies Adam. Returns the batch loss.
float trainStep() {
  zeroGrad();
  float lossSum = 0.0f;
  float invBatch = 1.0f / (float)BATCH_SIZE;

  for (uint16_t b = 0; b < BATCH_SIZE; b++) {
    uint16_t idx = (uint16_t)(xorshift32() % gReplaySize);
    const Transition &t = gReplay[idx];

    float y = t.r;
    if (!t.done) {
      forward(gTarget, t.s2, gAct);
      y += GAMMA * maxQ(gAct.q);
    }

    forward(gParams, t.s, gAct);
    float diff = gAct.q[t.a] - y;
    lossSum += diff * diff;

    float dq[N_ACTIONS];
    for (int a = 0; a < N_ACTIONS; a++) dq[a] = 0.0f;
    dq[t.a] = 2.0f * diff * invBatch;   // d(mean MSE)/dq
    backward(gAct, dq);
  }

  adamStep();
  return lossSum * invBatch;
}

float epsilonAt(uint16_t episode) {
  float frac = (float)episode / (float)EPS_DECAY_EPISODES;
  if (frac > 1.0f) frac = 1.0f;
  return EPS_START + frac * (EPS_END - EPS_START);
}

// ---------------------- Evaluation (greedy) ----------------------
struct EvalResult {
  float meanReturn;
  float stdReturn;
  float solvedRate;
  float avgInferenceUs;
};

EvalResult evaluate(const float *p, uint16_t nEpisodes) {
  float sum = 0.0f, sumSq = 0.0f;
  uint32_t solved = 0;
  uint64_t infUs = 0;
  uint32_t infCount = 0;

  for (uint16_t ep = 0; ep < nEpisodes; ep++) {
    envReset();
    bool done = false, terminated = false;
    float total = 0.0f;
    float s[OBS_DIM];
    while (!done) {
      yield();
      envState(s);
      uint32_t t0 = micros();
      forward(p, s, gAct);
      uint8_t a = argmaxQ(gAct.q);
      infUs += (uint32_t)(micros() - t0);
      infCount++;
      total += envStep(a, terminated, done);
    }
    sum += total;
    sumSq += total * total;
    if (total >= (float)MAX_STEPS) solved++;
  }

  EvalResult r;
  float n = (float)nEpisodes;
  r.meanReturn = sum / n;
  float var = sumSq / n - r.meanReturn * r.meanReturn;
  r.stdReturn = sqrtf(var > 0.0f ? var : 0.0f);
  r.solvedRate = (float)solved / n;
  r.avgInferenceUs = (float)((double)infUs / (double)infCount);
  return r;
}

void printEval(const EvalResult &r) {
  Serial.println(F("mean_return,std_return,solved_rate,avg_inference_us"));
  Serial.print(r.meanReturn, 2);     Serial.print(',');
  Serial.print(r.stdReturn, 2);      Serial.print(',');
  Serial.print(r.solvedRate, 4);     Serial.print(',');
  Serial.println(r.avgInferenceUs, 2);
}

// ---------------------- Reporting ----------------------
void printMemoryReport() {
  uint32_t learnerState = 6UL * (uint32_t)N_PARAMS * sizeof(float);
  uint32_t replayBytes  = (uint32_t)sizeof(gReplay);

  Serial.println(F("--- static RAM breakdown (on-device TRAINING) ---"));
  Serial.print(F("parameters per network: ")); Serial.println(N_PARAMS);
  Serial.print(F("one weight set (bytes): ")); Serial.println((uint32_t)sizeof(gParams));
  Serial.print(F("learner state, 6 weight-sized arrays (bytes): ")); Serial.println(learnerState);
  Serial.print(F("  = policy + target + best_checkpoint + grad + adam_m + adam_v"));
  Serial.println();
  Serial.print(F("sizeof(Transition) (bytes): ")); Serial.println((uint32_t)sizeof(Transition));
  Serial.print(F("replay capacity: ")); Serial.println(REPLAY_CAPACITY);
  Serial.print(F("replay buffer (bytes): ")); Serial.println(replayBytes);
  Serial.print(F("TOTAL training-only static RAM (bytes): "));
  Serial.println(learnerState + replayBytes);
  Serial.print(F("for reference, PC replay capacity 20000 would need (bytes): "));
  Serial.println(20000UL * (uint32_t)sizeof(Transition));
}

bool numericsReport() {
  float maxAbs = 0.0f;
  bool ok = true;
  for (int i = 0; i < N_PARAMS; i++) {
    float v = gParams[i];
    if (isBadFloat(v)) ok = false;
    float a = fabsf(v);
    if (a > maxAbs) maxAbs = a;
  }
  Serial.println(F("--- float32 numerical stability ---"));
  Serial.print(F("max_abs_weight: ")); Serial.println(maxAbs, 6);
  Serial.print(F("all_finite: ")); Serial.println(ok ? F("yes") : F("NO -- diverged"));
  return ok;
}

#if PRINT_FINAL_WEIGHTS
void printWeightArray(const char *name, const float *p, int offset, int count) {
  Serial.print(F("const float "));
  Serial.print(name);
  Serial.print('[');
  Serial.print(count);
  Serial.println(F("] PROGMEM_ATTR = {"));
  for (int i = 0; i < count; i++) {
    Serial.print(p[offset + i], 8);
    Serial.print('f');
    if (i != count - 1) Serial.print(',');
    if ((i % 8) == 7) Serial.println();
  }
  Serial.println(F("\n};"));
}

void dumpWeightsAsHeader(const float *p) {
  Serial.println();
  Serial.println(F("--- BEGIN cartpole_dqn_weights_onboard.h ---"));
  Serial.println(F("// Trained entirely on the microcontroller by cartpole_dqn_train.ino."));
  Serial.println(F("// Drop-in replacement for the PC-trained cartpole_dqn_weights.h."));
  Serial.println(F("#pragma once"));
  Serial.println();
  Serial.println(F("const int cartpole_n_layers = 3;"));
  Serial.print(F("const int cartpole_layer0_in = ")); Serial.print(OBS_DIM); Serial.println(';');
  Serial.print(F("const int cartpole_layer0_out = ")); Serial.print(H1); Serial.println(';');
  printWeightArray("cartpole_layer0_W", p, OFF_W0, H1 * OBS_DIM);
  printWeightArray("cartpole_layer0_b", p, OFF_B0, H1);
  Serial.print(F("const int cartpole_layer1_in = ")); Serial.print(H1); Serial.println(';');
  Serial.print(F("const int cartpole_layer1_out = ")); Serial.print(H2); Serial.println(';');
  printWeightArray("cartpole_layer1_W", p, OFF_W1, H2 * H1);
  printWeightArray("cartpole_layer1_b", p, OFF_B1, H2);
  Serial.print(F("const int cartpole_layer2_in = ")); Serial.print(H2); Serial.println(';');
  Serial.print(F("const int cartpole_layer2_out = ")); Serial.print(N_ACTIONS); Serial.println(';');
  printWeightArray("cartpole_layer2_W", p, OFF_W2, N_ACTIONS * H2);
  printWeightArray("cartpole_layer2_b", p, OFF_B2, N_ACTIONS);
  Serial.println(F("--- END cartpole_dqn_weights_onboard.h ---"));
}
#endif

void printBanner() {
  Serial.println();
  Serial.println(F("=== Cart-Pole GENUINE ON-DEVICE DQN TRAINING (backprop on the MCU) ==="));
  Serial.print(F("board: ")); Serial.println(boardName());
  Serial.print(F("free RAM at startup (bytes): ")); Serial.println(freeRamBytes());
  Serial.print(F("network: ")); Serial.print(OBS_DIM); Serial.print(F(" -> "));
  Serial.print(H1); Serial.print(F(" -> ")); Serial.print(H2);
  Serial.print(F(" -> ")); Serial.println(N_ACTIONS);
  printMemoryReport();
  Serial.println(F("--- configuration ---"));
  Serial.print(F("episodes: ")); Serial.print(N_EPISODES);
  Serial.print(F("  batch: ")); Serial.print(BATCH_SIZE);
  Serial.print(F("  lr: ")); Serial.print(LR, 6);
  Serial.print(F("  gamma: ")); Serial.println(GAMMA, 4);
  Serial.print(F("replay_capacity: ")); Serial.print(REPLAY_CAPACITY);
  Serial.print(F("  (PC used 20000)   min_replay: ")); Serial.println(MIN_REPLAY_BEFORE_TRAIN);
  Serial.println();
}

// ---------------------- Training run ----------------------
struct TrainStats {
  float    bestProbe = -1e30f;
  uint16_t bestProbeEpisode = 0;
  uint64_t trainStepUsTotal = 0;
  uint32_t trainStepCount = 0;
  uint64_t greedyFwdUsTotal = 0;
  uint32_t greedyFwdCount = 0;
  uint32_t globalStep = 0;
  uint32_t wallMs = 0;
  bool     diverged = false;
};
TrainStats gStats;

void runTraining() {
  double lossAccum = 0.0;
  uint32_t lossCount = 0;
  float returnAccum = 0.0f;

  Serial.println(F("episode,avg_return,eps,avg_loss,avg_train_step_us,free_ram_b"));

  uint32_t wallStart = millis();

  for (uint16_t ep = 0; ep < N_EPISODES; ep++) {
    envReset();
    float s[OBS_DIM], s2[OBS_DIM];
    envState(s);
    float eps = epsilonAt(ep);
    float totalReward = 0.0f;
    bool done = false, terminated = false;

    while (!done) {
      yield();   // keeps ESP watchdogs happy

      uint8_t a;
      if (randUnit() < eps) {
        a = (uint8_t)(xorshift32() % N_ACTIONS);
      } else {
        uint32_t t0 = micros();
        forward(gParams, s, gAct);
        a = argmaxQ(gAct.q);
        gStats.greedyFwdUsTotal += (uint32_t)(micros() - t0);
        gStats.greedyFwdCount++;
      }

      float r = envStep(a, terminated, done);
      envState(s2);
      replayPush(s, a, r, s2, terminated);   // bootstrap on truncation, not on terminal
      memcpy(s, s2, sizeof(s));
      totalReward += r;
      gStats.globalStep++;

      if (gReplaySize >= MIN_REPLAY_BEFORE_TRAIN) {
        uint32_t t0 = micros();
        float loss = trainStep();
        gStats.trainStepUsTotal += (uint32_t)(micros() - t0);
        gStats.trainStepCount++;
        lossAccum += (double)loss;
        lossCount++;

        if (gStats.globalStep % TARGET_UPDATE_EVERY_STEPS == 0)
          memcpy(gTarget, gParams, sizeof(gParams));
      }
    }

    returnAccum += totalReward;

    if ((ep + 1) % LOG_INTERVAL == 0) {
      Serial.print(ep + 1); Serial.print(',');
      Serial.print(returnAccum / (float)LOG_INTERVAL, 2); Serial.print(',');
      Serial.print(eps, 3); Serial.print(',');
      Serial.print(lossCount ? (float)(lossAccum / lossCount) : 0.0f, 4); Serial.print(',');
      Serial.print(gStats.trainStepCount
                   ? (float)((double)gStats.trainStepUsTotal / gStats.trainStepCount) : 0.0f, 1);
      Serial.print(',');
      Serial.println(freeRamBytes());
      returnAccum = 0.0f;
      lossAccum = 0.0; lossCount = 0;
    }

    // Checkpoint selection: keep the best network seen by a short greedy probe
    if ((ep + 1) % PROBE_INTERVAL == 0) {
      EvalResult probe = evaluate(gParams, PROBE_EPISODES);
      if (probe.meanReturn > gStats.bestProbe) {
        gStats.bestProbe = probe.meanReturn;
        gStats.bestProbeEpisode = (uint16_t)(ep + 1);
        memcpy(gBest, gParams, sizeof(gParams));
        Serial.print(F("  -> new best checkpoint at episode "));
        Serial.print(ep + 1);
        Serial.print(F(": greedy probe avg_return="));
        Serial.println(probe.meanReturn, 2);
      }
      // abort early if float32 training has blown up
      for (int i = 0; i < N_PARAMS && !gStats.diverged; i++)
        if (isBadFloat(gParams[i])) gStats.diverged = true;
      if (gStats.diverged) {
        Serial.print(F("!! training diverged (NaN/Inf in weights) at episode "));
        Serial.println(ep + 1);
        break;
      }
    }
  }

  gStats.wallMs = millis() - wallStart;
}

void printTrainingReport() {
  const TrainStats &st = gStats;

  Serial.println();
  Serial.println(F("--- on-device training cost ---"));
  Serial.print(F("total training wall-clock: ")); Serial.print(st.wallMs / 1000.0f, 1);
  Serial.print(F(" s (")); Serial.print(st.wallMs / 60000.0f, 2); Serial.println(F(" min)"));
  Serial.print(F("environment steps: ")); Serial.println(st.globalStep);
  Serial.print(F("gradient updates: ")); Serial.println(st.trainStepCount);
  Serial.print(F("avg full training step (batch fwd+bwd+Adam) us: "));
  Serial.println(st.trainStepCount ? (float)((double)st.trainStepUsTotal / st.trainStepCount) : 0.0f, 2);
  Serial.print(F("avg greedy forward pass us (comparable to inference sketch): "));
  Serial.println(st.greedyFwdCount ? (float)((double)st.greedyFwdUsTotal / st.greedyFwdCount) : 0.0f, 2);
  Serial.print(F("free RAM after training (bytes): ")); Serial.println(freeRamBytes());
  Serial.println();

  numericsReport();
  Serial.println();

  Serial.print(F("best checkpoint: episode ")); Serial.print(st.bestProbeEpisode);
  Serial.print(F(", greedy probe avg_return=")); Serial.println(st.bestProbe, 2);
  Serial.println();

  Serial.println(F("=== final evaluation, BEST checkpoint (100 greedy episodes) ==="));
  printEval(evaluate(gBest, N_EVAL_EPISODES));

  Serial.println(F("=== final evaluation, LAST-episode weights (for comparison) ==="));
  printEval(evaluate(gParams, N_EVAL_EPISODES));

#if PRINT_FINAL_WEIGHTS
  dumpWeightsAsHeader(gBest);
#endif

  Serial.println();
  Serial.println(F("=== done: this policy was learned entirely on the microcontroller ==="));
}

// ---------------------- Arduino entry points ----------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  delay(1500);   // give the Serial Monitor time to attach

  printBanner();

  rngState = SEED;
  if (rngState == 0) rngState = 0xA5A5A5A5;
  initNetwork();

  runTraining();
  printTrainingReport();
}

void loop() { }
