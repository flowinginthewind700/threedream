/**
 * "Drive" task: steer a body to a randomly placed goal under real dynamics.
 *
 * Contact-free on purpose. This is the reference task for the trainer: it has a
 * second-order plant (forces integrate into velocity, velocity into position, and
 * linear damping bleeds both off), a randomised goal per episode, and a reward
 * that is exactly zero-sum under potential shaping. The bundled policy-gradient
 * trainer reaches ~98% greedy success on it in under 2000 episodes, so it is the
 * fast regression test for "is the learning stack still correct".
 *
 * `ReachEnv` is the harder sibling: same loop, but the agent must *push* a puck
 * through contact, which is a much narrower exploration corridor.
 */

import { Rng } from '../core/rng.js';
import { BuiltinPhysics } from '../physics/builtin.js';
import {
  distanceVec3,
  lengthVec3,
  subVec3,
  vec3,
  type PhysicsBackend,
  type Vec3,
} from '../physics/types.js';
import type { LearningEnvironment, RewardShape, StepResult } from './types.js';

export interface DriveEnvOptions {
  backend?: PhysicsBackend;
  fixedDt?: number;
  maxStepsPerEpisode?: number;
  discount?: number;
  /** Max planar force the action can command. */
  maxForce?: number;
  /** Goal counts as reached within this distance. */
  goalTolerance?: number;
  /** Half-size of the square spawn region for agent and goal. */
  spawnHalf?: number;
  reward?: RewardShape;
  seed?: number;
}

export interface DriveDiagnostics {
  readonly position: Vec3;
  readonly velocity: Vec3;
  readonly goalPosition: Vec3;
  readonly distance: number;
  readonly reached: boolean;
}

export class DriveEnv implements LearningEnvironment {
  readonly name = 'drive';
  readonly observationSize = 6;
  readonly actionSize = 2;
  readonly discount: number;
  readonly maxStepsPerEpisode: number;

  readonly backend: PhysicsBackend;
  readonly goalTolerance: number;
  readonly maxForce: number;

  private readonly ownsBackend: boolean;
  private readonly fixedDt: number;
  private readonly spawnHalf: number;
  private readonly rewardWeights: Required<Pick<RewardShape, 'reach' | 'effort' | 'step' | 'success'>>;
  private readonly observation = new Float32Array(this.observationSize);

  private rng: Rng;
  private body = 0;
  private goal: Vec3 = vec3();
  private stepsThisEpisode = 0;
  private lastPotential = 0;
  private reached = false;

  constructor(options: DriveEnvOptions = {}) {
    this.fixedDt = options.fixedDt ?? 1 / 60;
    this.ownsBackend = options.backend === undefined;
    this.backend =
      options.backend ??
      new BuiltinPhysics({
        fixedDt: this.fixedDt,
        gravity: vec3(0, 0, 0), // top-down plane
        linearDamping: 2.0,
        angularDamping: 2.0,
        solverIterations: 4,
      });
    this.discount = options.discount ?? 0.99;
    this.maxStepsPerEpisode = options.maxStepsPerEpisode ?? 100;
    this.maxForce = options.maxForce ?? 5;
    this.goalTolerance = options.goalTolerance ?? 0.15;
    this.spawnHalf = options.spawnHalf ?? 0.7;
    this.rewardWeights = {
      reach: options.reward?.reach ?? 3,
      success: options.reward?.success ?? 10,
      effort: options.reward?.effort ?? 0.001,
      step: options.reward?.step ?? 0.005,
    };
    this.rng = new Rng(options.seed ?? 11);
    this.body = this.backend.createBody({
      shape: { kind: 'sphere', radius: 0.08 },
      position: vec3(),
      mass: 1,
      friction: 0.5,
      label: 'agent',
    });
    this.reset(this.rng);
  }

  reset(rng: Rng): Float32Array {
    this.rng = rng;
    this.stepsThisEpisode = 0;
    this.reached = false;
    const h = this.spawnHalf;
    this.goal = vec3(rng.range(-h, h), 0, rng.range(-h, h));
    this.backend.setBodyState(this.body, {
      position: vec3(rng.range(-h, h), 0, rng.range(-h, h)),
      velocity: vec3(),
      angularVelocity: vec3(),
    });
    this.backend.drainContacts();
    this.lastPotential = this.potential();
    return this.observe(this.observation);
  }

  /**
   * Phi(s) = -distance to goal, the shaping potential. See `ReachEnv.potential`
   * for the full argument on why potential-based shaping beats a hand-tuned
   * dense bonus here.
   */
  private potential(): number {
    return -distanceVec3(this.position(), this.goal);
  }

  step(action: ArrayLike<number>): StepResult {
    const ax = clamp(action[0] ?? 0, -1, 1);
    const az = clamp(action[1] ?? 0, -1, 1);
    const f = this.maxForce;
    this.backend.applyImpulse(this.body, vec3((ax * f * this.fixedDt), 0, az * f * this.fixedDt));
    this.backend.step(this.fixedDt);
    this.backend.drainContacts();
    this.stepsThisEpisode++;

    const potential = this.potential();
    // Undiscounted difference. Ng et al. use gamma*Phi(s') - Phi(s); dropping the
    // gamma leaves an extra (1-gamma)*Phi(s') term, which at gamma=0.99 is 1% of
    // the potential and reads as a mild pull toward the goal rather than noise.
    // Both envs use the same form so their rewards stay comparable.
    const shaping = potential - this.lastPotential;
    this.lastPotential = potential;

    const distance = -potential;
    const justReached = !this.reached && distance <= this.goalTolerance;
    if (justReached) this.reached = true;

    const w = this.rewardWeights;
    let reward = -w.step + w.reach * shaping - w.effort * (Math.abs(ax) + Math.abs(az));
    if (justReached) reward += w.success;

    return {
      reward,
      done: this.reached || this.stepsThisEpisode >= this.maxStepsPerEpisode,
      terminated: this.reached,
      info: { distance, shaping, steps: this.stepsThisEpisode },
    };
  }

  observe(out: Float32Array): Float32Array {
    const state = this.backend.getBodyState(this.body);
    const position = state?.position ?? vec3();
    const velocity = state?.velocity ?? vec3();
    const toGoal = subVec3(this.goal, position);
    const distance = lengthVec3(toGoal);
    let i = 0;
    out[i++] = position[0];
    out[i++] = position[2];
    out[i++] = distance > 1e-6 ? toGoal[0] / distance : 0;
    out[i++] = distance > 1e-6 ? toGoal[2] / distance : 0;
    out[i++] = distance;
    out[i++] = Math.min(1, lengthVec3(velocity) * 0.3);
    return out;
  }

  diagnostics(): DriveDiagnostics {
    return {
      position: this.position(),
      velocity: this.backend.getBodyState(this.body)?.velocity ?? vec3(),
      goalPosition: this.goal,
      distance: distanceVec3(this.position(), this.goal),
      reached: this.reached,
    };
  }

  get handle(): number {
    return this.body;
  }

  private position(): Vec3 {
    return this.backend.getBodyState(this.body)?.position ?? vec3();
  }

  dispose(): void {
    if (this.ownsBackend) this.backend.dispose();
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
