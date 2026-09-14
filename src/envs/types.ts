/**
 * Learning environment contract.
 *
 * Mirrors the gymnasium-style loop (reset -> step -> observe) but with typed
 * fixed-size buffers, because allocation inside a training loop dominates cost
 * once episodes run in the thousands. Implementations own their own physics
 * backend, so an env can run headless in Node or be rendered live in the
 * browser without changing the trainer.
 */

import type { Rng } from '../core/rng.js';

export interface StepResult {
  reward: number;
  done: boolean;
  /**
   * True only when the episode ended for a *real* reason (goal reached,
   * failure), as opposed to hitting the step budget. Trainers use it to decide
   * whether to bootstrap `V(s_{t+1})` at the final transition: a time limit is
   * an artifact of the harness, not a property of the task, so treating it as
   * terminal teaches the critic that value decays to zero as the clock runs
   * out. Omitted means "same as `done`".
   */
  terminated?: boolean;
  /** Optional diagnostics, surfaced in training logs. */
  info?: Record<string, number>;
}

export interface LearningEnvironment {
  readonly name: string;
  readonly observationSize: number;
  readonly actionSize: number;
  /** Discount factor gamma in [0, 1). */
  readonly discount: number;
  readonly maxStepsPerEpisode: number;

  /** Reset and return the first observation (fresh array, owned by the env). */
  reset(rng: Rng): Float32Array;

  /** Apply an action for one fixed step. */
  step(action: ArrayLike<number>): StepResult;

  /** Write the current observation into `out` and return it. */
  observe(out: Float32Array): Float32Array;

  /** Release physics resources. */
  dispose(): void;
}

export interface RewardShape {
  /**
   * Weight on the potential-based shaping term, which combines the puck-goal
   * gap and the agent-puck gap. See `ReachEnv` for why this is shaping rather
   * than a hand-tuned dense bonus.
   */
  reach?: number;
  /** Weight on the bonus for being within goal tolerance. */
  success?: number;
  /** Weight penalising total action magnitude (smoothness). */
  effort?: number;
  /** Distance at which the goal counts as reached. */
  goalTolerance?: number;
  /**
   * Relative weight of the agent-puck gap inside the shaping potential.
   * 0 makes the potential puck-to-goal only; 1 weights both gaps equally.
   */
  approach?: number;
  /** Flat per-step living cost. Keeps idling worse than acting. */
  step?: number;
}
