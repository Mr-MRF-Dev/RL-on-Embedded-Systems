/*
  Grid-World on-device Tabular Q-Learning

  Environment simulation and Q-learning both run on the microcontroller.
  Plain Arduino-core C++: Uno/Nano, ESP8266, ESP32, STM32 (STM32duino).
  Serial: 115200 baud.

  Memory scaling study: the Q-table takes GRID_N^2 * 4 actions * 4 bytes.
  Change GRID_N and recompile; the compiler's RAM usage report (or a RAM
  overflow at link time) shows where a board runs out of memory.

  Output (CSV):
    - GRID_N, N_STATES and Q-table size in bytes (printed once)
    - training log: episode,return,steps,avg_step_us,epsilon
    - post-training greedy evaluation summary (mean/std over N_EVAL_EPISODES)
*/

#include <math.h>

// ---------------------- Configuration ----------------------
const uint16_t GRID_N = 6;   // grid is GRID_N x GRID_N

// 0 = core budget (2000 episodes), 1 = extended budget (30000 episodes, 15x)
#define EXTENDED_BUDGET 0

// ---------------------- Problem definition (matches pc/env.py) ----------------------
const uint16_t N_STATES  = GRID_N * GRID_N;
const uint8_t  N_ACTIONS = 4;   // 0 = Up, 1 = Down, 2 = Left, 3 = Right

const float STEP_COST         = -1.0f;
const float WALL_BUMP_PENALTY = -2.0f;
const float GOAL_REWARD       = 50.0f;
const uint16_t MAX_STEPS      = (6 * GRID_N > 50) ? (6 * GRID_N) : 50;

// ---------------------- Q-learning hyperparameters ----------------------
const float    ALPHA     = 0.15f;
const float    GAMMA     = 0.95f;
const float    EPS_START = 1.0f;
const float    EPS_END   = 0.05f;
const uint32_t N_EVAL_EPISODES = 100;

#if EXTENDED_BUDGET
const uint16_t EPS_DECAY_EPISODES = 12000;
const uint32_t N_EPISODES         = 30000;
const uint32_t LOG_INTERVAL       = 1000;
#else
const uint16_t EPS_DECAY_EPISODES = 800;
const uint32_t N_EPISODES         = 2000;
const uint32_t LOG_INTERVAL       = 50;
#endif

// ---------------------- Global state ----------------------
float Q[N_STATES][N_ACTIONS];

// environment
int16_t  agentRow, agentCol;
uint16_t stepCount;
bool     goalReached;

// training
uint32_t episode = 0;
uint64_t totalStepTimeUs = 0;
uint32_t totalStepsTimed = 0;

// evaluation
uint32_t evalEpisode = 0;
uint32_t evalGoalReached = 0;
bool     evalStarted = false;
bool     reported    = false;
float    evalReturnSum = 0.0f, evalReturnSumSq = 0.0f;
float    evalStepsSum  = 0.0f, evalStepsSumSq  = 0.0f;

// ---------------------- PRNG (xorshift32) ----------------------
uint32_t rngState;

uint32_t xorshift32() {
  rngState ^= rngState << 13;
  rngState ^= rngState >> 17;
  rngState ^= rngState << 5;
  return rngState;
}

float randUnit()      { return (xorshift32() >> 8) / 16777216.0f; }  // [0, 1)
uint8_t randAction()  { return xorshift32() % N_ACTIONS; }

// ---------------------- Environment ----------------------
uint16_t getState() {
  return (uint16_t)agentRow * GRID_N + (uint16_t)agentCol;
}

void envReset() {
  agentRow    = 0;
  agentCol    = 0;
  stepCount   = 0;
  goalReached = false;
}

float envStep(uint8_t action, uint16_t &nextState, bool &done) {
  stepCount++;
  int16_t oldRow = agentRow, oldCol = agentCol;

  if      (action == 0) agentRow -= 1;
  else if (action == 1) agentRow += 1;
  else if (action == 2) agentCol -= 1;
  else                  agentCol += 1;

  bool wallBump = false;
  if (agentRow < 0 || agentRow >= (int16_t)GRID_N ||
      agentCol < 0 || agentCol >= (int16_t)GRID_N) {
    agentRow = oldRow;
    agentCol = oldCol;
    wallBump = true;
  }

  float reward = STEP_COST;
  if (wallBump) reward += WALL_BUMP_PENALTY;

  done = false;
  if (agentRow == (int16_t)(GRID_N - 1) && agentCol == (int16_t)(GRID_N - 1)) {
    reward      = GOAL_REWARD;
    goalReached = true;
    done        = true;
  }
  if (stepCount >= MAX_STEPS) done = true;

  nextState = getState();
  return reward;
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
    Serial.print(stepCount);   Serial.print(',');
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
  evalStepsSum    += stepCount;
  evalStepsSumSq  += (float)stepCount * (float)stepCount;
  if (goalReached) evalGoalReached++;
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
  Serial.println(F("mean_return,std_return,mean_steps,std_steps,goal_reach_rate,n_eval_episodes"));
  Serial.print(evalReturnSum / n);                          Serial.print(',');
  Serial.print(stdDev(evalReturnSum, evalReturnSumSq, n));  Serial.print(',');
  Serial.print(evalStepsSum / n);                           Serial.print(',');
  Serial.print(stdDev(evalStepsSum, evalStepsSumSq, n));    Serial.print(',');
  Serial.print((float)evalGoalReached / n, 4);              Serial.print(',');
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

  Serial.println(F("=== Grid-World on-device Tabular Q-Learning ==="));
  Serial.print(F("GRID_N: "));   Serial.println(GRID_N);
  Serial.print(F("N_STATES: ")); Serial.println(N_STATES);
  Serial.print(F("Q-table memory footprint (bytes): "));
  Serial.println(sizeof(Q));
  Serial.println(F("episode,return,steps,avg_step_us,epsilon"));
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
