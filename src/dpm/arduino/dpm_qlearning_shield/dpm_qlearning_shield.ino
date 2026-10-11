/*
  DPM on-device Tabular Q-Learning with a two-tier Safety Shield

  Same environment and Q-learning as dpm_qlearning.ino, plus a rule-based
  shield between the agent and the environment (active in training and
  evaluation):
    - Hard tier   (battery <= SHIELD_THRESHOLD): the proposed action is
      overridden with Sleep.
    - Warning tier (battery < WARNING_THRESHOLD): no override; a reward
      penalty that grows as the battery drops is applied instead.
  Q-updates use the executed action (after any override).

  Plain Arduino-core C++: Uno/Nano, ESP8266, ESP32, STM32 (STM32duino).
  Serial: 115200 baud.

  Output (CSV):
    - shield thresholds and Q-table size in bytes (printed once)
    - training log every 50 episodes: episode,return,survival_steps,avg_step_us,epsilon
    - shield interventions and battery-depletion episodes during training
    - post-training greedy evaluation summary (shield active)
    - learned greedy policy per (battery_bin, idle_bin)

  The battery level is simulated; no real energy is measured.
*/

#include <math.h>

// ---------------------- Problem definition (matches pc/env.py) ----------------------
const uint8_t N_BATTERY_BINS = 5;   // 0 = critical ... 4 = full
const uint8_t N_IDLE_BINS    = 4;   // steps since last request, capped at 3
const uint8_t N_STATES       = N_BATTERY_BINS * N_IDLE_BINS;
const uint8_t N_ACTIONS      = 3;   // 0 = Active, 1 = Idle, 2 = Sleep
const uint8_t ACTION_SLEEP   = 2;

const float POWER_COST[N_ACTIONS]      = {3.0f, 1.0f, 0.2f};
const float LATENCY_PENALTY[N_ACTIONS] = {0.0f, 1.0f, 3.0f};
const float BASELINE_LEAK    = 0.1f;
const float REQUEST_PROB     = 0.30f;
const float REWARD_SCALE     = 3.0f;
const float SURVIVAL_BONUS   = 1.5f;
const float TERMINAL_PENALTY = -50.0f;
const uint16_t MAX_STEPS     = 200;

// ---------------------- Safety Shield parameters ----------------------
const float SHIELD_THRESHOLD  = 15.0f;  // hard tier: force Sleep at or below this level
const float WARNING_THRESHOLD = 30.0f;  // warning tier: reward penalty below this level
const float WARN_WEIGHT       = 4.0f;   // max warning penalty (reached at battery = 0)

// ---------------------- Q-learning hyperparameters ----------------------
const float    ALPHA              = 0.1f;
const float    GAMMA              = 0.99f;
const float    EPS_START          = 1.0f;
const float    EPS_END            = 0.05f;
const uint16_t EPS_DECAY_EPISODES = 2500;
const uint32_t N_EPISODES         = 6000;
const uint32_t N_EVAL_EPISODES    = 100;
const uint16_t LOG_INTERVAL       = 50;

// ---------------------- Global state ----------------------
float Q[N_STATES][N_ACTIONS];

// environment
float    battery;
uint8_t  idleSteps;
uint16_t stepCount;

// training
uint32_t episode = 0;
uint64_t totalStepTimeUs = 0;
uint32_t totalStepsTimed = 0;
uint32_t shieldInterventions = 0;   // steps where the hard tier overrode the action
uint32_t trainingDepletions  = 0;   // training episodes that ended with an empty battery

// evaluation
uint32_t evalEpisode = 0;
bool     evalStarted = false;
bool     reported    = false;
float    evalReturnSum = 0.0f,   evalReturnSumSq = 0.0f;
float    evalSurvivalSum = 0.0f, evalSurvivalSumSq = 0.0f;

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

// ---------------------- State discretization ----------------------
uint8_t batteryBin() {
  float level = battery;
  if (level < 0)   level = 0;
  if (level > 100) level = 100;
  int bin = (int)(level / 20.0f);
  if (bin > N_BATTERY_BINS - 1) bin = N_BATTERY_BINS - 1;
  return (uint8_t)bin;
}

uint8_t idleBin() {
  return idleSteps > (N_IDLE_BINS - 1) ? (N_IDLE_BINS - 1) : idleSteps;
}

uint8_t getState() {
  return batteryBin() * N_IDLE_BINS + idleBin();
}

// ---------------------- Environment ----------------------
void envReset() {
  battery   = 100.0f;
  idleSteps = 0;
  stepCount = 0;
}

float envStep(uint8_t action, uint8_t &nextState, bool &done) {
  stepCount++;

  float cost = POWER_COST[action] + BASELINE_LEAK;
  battery -= cost;

  float latency = 0.0f;
  if (randUnit() < REQUEST_PROB) {
    latency   = LATENCY_PENALTY[action];
    idleSteps = 0;
  } else {
    idleSteps++;
  }

  float reward = SURVIVAL_BONUS - (cost / REWARD_SCALE) - latency;

  done = false;
  if (battery <= 0.0f) {
    reward += TERMINAL_PENALTY;
    done = true;
  }
  if (stepCount >= MAX_STEPS) done = true;

  nextState = getState();
  return reward;
}

// ---------------------- Safety Shield ----------------------
// Returns the action actually executed.
uint8_t shield(float batteryLevel, uint8_t proposedAction) {
  if (batteryLevel <= SHIELD_THRESHOLD && proposedAction != ACTION_SLEEP) {
    return ACTION_SLEEP;
  }
  return proposedAction;
}

float warningPenalty(float batteryLevel) {
  if (batteryLevel < WARNING_THRESHOLD) {
    return WARN_WEIGHT * (WARNING_THRESHOLD - batteryLevel) / WARNING_THRESHOLD;
  }
  return 0.0f;
}

// ---------------------- Q-learning ----------------------
uint8_t greedyAction(uint8_t s) {
  uint8_t best = 0;
  for (uint8_t a = 1; a < N_ACTIONS; a++) {
    if (Q[s][a] > Q[s][best]) best = a;
  }
  return best;
}

float maxQ(uint8_t s) {
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
  uint8_t s = getState();

  while (!done) {
    uint32_t t0 = micros();

    uint8_t proposed = (randUnit() < eps) ? randAction() : greedyAction(s);
    uint8_t a = shield(battery, proposed);
    if (a != proposed) shieldInterventions++;
    float batteryBefore = battery;

    uint8_t s2;
    float r = envStep(a, s2, done);
    r -= warningPenalty(batteryBefore);

    float target = done ? r : (r + GAMMA * maxQ(s2));
    Q[s][a] += ALPHA * (target - Q[s][a]);

    totalStepTimeUs += (micros() - t0);
    totalStepsTimed++;

    s = s2;
    totalReward += r;
  }

  if (battery <= 0.0f) trainingDepletions++;

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

// ---------------------- Evaluation (greedy, Q frozen, shield active) ----------------------
void runEvalEpisode() {
  envReset();
  float totalReward = 0.0f;
  bool done = false;
  uint8_t s = getState();

  while (!done) {
    uint8_t a = shield(battery, greedyAction(s));
    uint8_t s2;
    float r = envStep(a, s2, done);
    s = s2;
    totalReward += r;
  }

  evalReturnSum     += totalReward;
  evalReturnSumSq   += totalReward * totalReward;
  evalSurvivalSum   += stepCount;
  evalSurvivalSumSq += (float)stepCount * (float)stepCount;
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
  Serial.println(F("mean_return,std_return,mean_survival,std_survival,n_eval_episodes"));
  Serial.print(evalReturnSum / n);                              Serial.print(',');
  Serial.print(stdDev(evalReturnSum, evalReturnSumSq, n));      Serial.print(',');
  Serial.print(evalSurvivalSum / n);                            Serial.print(',');
  Serial.print(stdDev(evalSurvivalSum, evalSurvivalSumSq, n));  Serial.print(',');
  Serial.println(N_EVAL_EPISODES);
}

void printPolicy() {
  Serial.println(F("=== learned greedy policy ==="));
  Serial.println(F("battery_bin,idle_bin,action(0=A,1=I,2=S)"));
  for (uint8_t b = 0; b < N_BATTERY_BINS; b++) {
    for (uint8_t i = 0; i < N_IDLE_BINS; i++) {
      Serial.print(b); Serial.print(',');
      Serial.print(i); Serial.print(',');
      Serial.println(greedyAction(b * N_IDLE_BINS + i));
    }
  }
}

// ---------------------- Arduino entry points ----------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  // seed from a floating analog pin + timer
  rngState = analogRead(A0) * 2654435761u + micros();
  if (rngState == 0) rngState = 0xA5A5A5A5;

  for (uint8_t s = 0; s < N_STATES; s++)
    for (uint8_t a = 0; a < N_ACTIONS; a++)
      Q[s][a] = 0.0f;

  Serial.println(F("=== DPM + Safety Shield on-device Tabular Q-Learning ==="));
  Serial.print(F("SHIELD_THRESHOLD: "));  Serial.println(SHIELD_THRESHOLD);
  Serial.print(F("WARNING_THRESHOLD: ")); Serial.println(WARNING_THRESHOLD);
  Serial.print(F("Q-table memory footprint (bytes): "));
  Serial.println(sizeof(Q));
  Serial.println(F("episode,return,survival_steps,avg_step_us,epsilon"));
}

void loop() {
  if (episode < N_EPISODES) {
    runTrainingEpisode();
    return;
  }

  if (!evalStarted) {
    Serial.println(F("=== training complete: starting post-training evaluation ==="));
    Serial.print(F("shield interventions during training (hard tier fired): "));
    Serial.println(shieldInterventions);
    Serial.print(F("training episodes that ended via battery depletion (out of "));
    Serial.print(N_EPISODES); Serial.print(F("): "));
    Serial.println(trainingDepletions);
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
    printPolicy();
    reported = true;
  }
}
