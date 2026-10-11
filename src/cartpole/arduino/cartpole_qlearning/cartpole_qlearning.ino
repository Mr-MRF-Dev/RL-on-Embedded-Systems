/*
  Cart-Pole on-device Tabular Q-Learning

  Physics simulation and Q-learning both run on the microcontroller.
  Plain Arduino-core C++: Uno/Nano, ESP8266, ESP32, STM32 (STM32duino).
  Serial: 115200 baud.

  The 4-D continuous state is discretized into NUM_BINS bins per dimension,
  so the Q-table takes NUM_BINS^4 * 2 actions * 4 bytes. Change NUM_BINS and
  recompile; the compiler's RAM usage report (or a RAM overflow at link time)
  shows where a board runs out of memory.

  Output (CSV):
    - NUM_BINS, N_STATES and Q-table size in bytes (printed once)
    - training log: episode,return,avg_step_us,epsilon
    - post-training greedy evaluation summary (mean/std over N_EVAL_EPISODES)
*/

#include <math.h>

// ---------------------- Configuration ----------------------
const uint8_t NUM_BINS = 6;   // bins per dimension

// 0 = core budget (10000 episodes), 1 = extended budget (50000 episodes, 5x)
#define EXTENDED_BUDGET 0

// ---------------------- Problem definition (matches pc/env.py) ----------------------
const uint16_t N_STATES  = (uint16_t)NUM_BINS * NUM_BINS * NUM_BINS * NUM_BINS;
const uint8_t  N_ACTIONS = 2;   // 0 = push left, 1 = push right

// Physics (standard CartPole-v1)
const float GRAVITY         = 9.8f;
const float MASSCART        = 1.0f;
const float MASSPOLE        = 0.1f;
const float TOTAL_MASS      = MASSCART + MASSPOLE;
const float LENGTH          = 0.5f;   // half the pole length
const float POLEMASS_LENGTH = MASSPOLE * LENGTH;
const float FORCE_MAG       = 10.0f;
const float TAU             = 0.02f;  // seconds per step

// Termination
const float THETA_THRESHOLD = 12.0f * 2.0f * 3.14159265f / 360.0f;
const float X_THRESHOLD     = 2.4f;
const uint16_t MAX_STEPS    = 200;

// Discretization bounds
const float X_LO        = -2.4f, X_HI        = 2.4f;
const float XDOT_LO     = -3.0f, XDOT_HI     = 3.0f;
const float THETADOT_LO = -3.5f, THETADOT_HI = 3.5f;
// theta bounds are +-THETA_THRESHOLD

// ---------------------- Q-learning hyperparameters ----------------------
const float    ALPHA     = 0.1f;
const float    GAMMA     = 0.99f;
const float    EPS_START = 1.0f;
const float    EPS_END   = 0.05f;
const uint32_t N_EVAL_EPISODES = 100;

#if EXTENDED_BUDGET
const uint16_t EPS_DECAY_EPISODES = 20000;
const uint32_t N_EPISODES         = 50000;
const uint32_t LOG_INTERVAL       = 2500;
#else
const uint16_t EPS_DECAY_EPISODES = 4000;
const uint32_t N_EPISODES         = 10000;
const uint32_t LOG_INTERVAL       = 500;
#endif

// ---------------------- Global state ----------------------
float Q[N_STATES][N_ACTIONS];

// environment
float    cartX, cartXDot, poleTheta, poleThetaDot;
uint16_t stepCount;

// training
uint32_t episode = 0;
uint64_t totalStepTimeUs = 0;
uint32_t totalStepsTimed = 0;

// evaluation
uint32_t evalEpisode = 0;
uint32_t evalSolvedCount = 0;
bool     evalStarted = false;
bool     reported    = false;
float    evalReturnSum = 0.0f, evalReturnSumSq = 0.0f;

// ---------------------- PRNG (xorshift32) ----------------------
uint32_t rngState;

uint32_t xorshift32() {
  rngState ^= rngState << 13;
  rngState ^= rngState >> 17;
  rngState ^= rngState << 5;
  return rngState;
}

float randUnit()      { return (xorshift32() >> 8) / 16777216.0f; }  // [0, 1)
float randSmall()     { return (randUnit() - 0.5f) * 0.1f; }         // [-0.05, 0.05)
uint8_t randAction()  { return xorshift32() % N_ACTIONS; }

// ---------------------- State discretization ----------------------
uint8_t binIndex(float value, float lo, float hi) {
  float v = value;
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  int idx = (int)((v - lo) / (hi - lo) * NUM_BINS);
  if (idx >= NUM_BINS) idx = NUM_BINS - 1;
  if (idx < 0)         idx = 0;
  return (uint8_t)idx;
}

uint16_t getState() {
  uint32_t n    = NUM_BINS;
  uint32_t bx   = binIndex(cartX,        X_LO,        X_HI);
  uint32_t bxd  = binIndex(cartXDot,     XDOT_LO,     XDOT_HI);
  uint32_t bth  = binIndex(poleTheta,    -THETA_THRESHOLD, THETA_THRESHOLD);
  uint32_t bthd = binIndex(poleThetaDot, THETADOT_LO, THETADOT_HI);
  return (uint16_t)((((bx * n + bxd) * n + bth) * n) + bthd);
}

// ---------------------- Environment ----------------------
void envReset() {
  cartX        = randSmall();
  cartXDot     = randSmall();
  poleTheta    = randSmall();
  poleThetaDot = randSmall();
  stepCount    = 0;
}

float envStep(uint8_t action, uint16_t &nextState, bool &done) {
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

  bool terminated = (cartX < -X_THRESHOLD || cartX > X_THRESHOLD ||
                     poleTheta < -THETA_THRESHOLD || poleTheta > THETA_THRESHOLD);
  bool truncated  = (stepCount >= MAX_STEPS);
  done = terminated || truncated;

  nextState = getState();
  return 1.0f;
}

// ---------------------- Q-learning ----------------------
uint8_t greedyAction(uint16_t s) {
  uint8_t best = 0;
  for (uint8_t a = 1; a < N_ACTIONS; a++) {
    if (Q[s][a] > Q[s][best]) best = a;
  }
  return best;
}

float maxQ(uint16_t s) {
  return Q[s][greedyAction(s)];
}

float epsilonAt(uint32_t ep) {
  float frac = (float)ep / EPS_DECAY_EPISODES;
  if (frac > 1.0f) frac = 1.0f;
  return EPS_START + frac * (EPS_END - EPS_START);
}

// ---------------------- Training ----------------------
void runTrainingEpisode() {
  envReset();
  float eps = epsilonAt(episode);
  float totalReward = 0.0f;
  bool done = false;
  uint16_t s = getState();

  while (!done) {
    uint32_t t0 = micros();

    uint8_t a = (randUnit() < eps) ? randAction() : greedyAction(s);

    uint16_t s2;
    float r = envStep(a, s2, done);

    float target = done ? r : (r + GAMMA * maxQ(s2));
    Q[s][a] += ALPHA * (target - Q[s][a]);

    totalStepTimeUs += (micros() - t0);
    totalStepsTimed++;

    s = s2;
    totalReward += r;
  }

  if (episode % LOG_INTERVAL == 0 || episode == N_EPISODES - 1) {
    float avgStepUs = (float)totalStepTimeUs / totalStepsTimed;
    Serial.print(episode);     Serial.print(',');
    Serial.print(totalReward); Serial.print(',');
    Serial.print(avgStepUs);   Serial.print(',');
    Serial.println(eps);
  }

  episode++;
}

// ---------------------- Evaluation (greedy, Q frozen) ----------------------
void runEvalEpisode() {
  envReset();
  float totalReward = 0.0f;
  bool done = false;
  uint16_t s = getState();

  while (!done) {
    uint16_t s2;
    float r = envStep(greedyAction(s), s2, done);
    s = s2;
    totalReward += r;
  }

  evalReturnSum   += totalReward;
  evalReturnSumSq += totalReward * totalReward;
  if (stepCount >= MAX_STEPS) evalSolvedCount++;
  evalEpisode++;
}

float stdDev(float sum, float sumSq, float n) {
  float m = sum / n;
  float variance = (sumSq / n) - (m * m);
  return sqrt(variance > 0.0f ? variance : 0.0f);
}

void printEvalSummary() {
  float n = (float)N_EVAL_EPISODES;

  Serial.println(F("=== evaluation summary (greedy policy, epsilon=0) ==="));
  Serial.println(F("mean_return,std_return,solved_rate,n_eval_episodes"));
  Serial.print(evalReturnSum / n);                          Serial.print(',');
  Serial.print(stdDev(evalReturnSum, evalReturnSumSq, n));  Serial.print(',');
  Serial.print((float)evalSolvedCount / n, 4);              Serial.print(',');
  Serial.println(N_EVAL_EPISODES);
}

// ---------------------- Arduino entry points ----------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  // seed from a floating analog pin + timer
  rngState = analogRead(A0) * 2654435761u + micros();
  if (rngState == 0) rngState = 0xA5A5A5A5;

  for (uint16_t s = 0; s < N_STATES; s++)
    for (uint8_t a = 0; a < N_ACTIONS; a++)
      Q[s][a] = 0.0f;

  Serial.println(F("=== Cart-Pole on-device Tabular Q-Learning ==="));
  Serial.print(F("NUM_BINS: ")); Serial.println(NUM_BINS);
  Serial.print(F("N_STATES: ")); Serial.println(N_STATES);
  Serial.print(F("Q-table memory footprint (bytes): "));
  Serial.println(sizeof(Q));
  Serial.println(F("episode,return,avg_step_us,epsilon"));
}

void loop() {
  if (episode < N_EPISODES) {
    runTrainingEpisode();
    return;
  }

  if (!evalStarted) {
    Serial.println(F("=== training complete: starting post-training evaluation ==="));
    Serial.print(F("eval episodes: "));
    Serial.println(N_EVAL_EPISODES);
    evalStarted = true;
  }

  if (evalEpisode < N_EVAL_EPISODES) {
    runEvalEpisode();
    return;
  }

  if (!reported) {
    printEvalSummary();
    Serial.println(F("=== done ==="));
    reported = true;
  }
}
