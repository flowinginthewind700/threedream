/**
 * "Reach" task: an agent body on a table must drive a puck to a randomly placed
 * goal using a 2D force action.
 *
 * This is the smallest task that exercises the whole stack: a real physics
 * backend, contact-driven dynamics (the agent has to actually push, not teleport),
 * a randomised goal for generalisation, and a dense reward that a REINFORCE
 * trainer can climb in a few thousand episodes. It is also the task the browser
 * demo renders, so what trains headless is what you see.
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

export interface ReachEnvOptions {
  backend?: PhysicsBackend;
  fixedDt?: number;
  maxStepsPerEpisode?: number;
  discount?: number;
  /** Half-size of the square table the puck is confined to. */
  arenaHalf?: number;
  goalTolerance?: number;
  reward?: RewardShape;
  /** Max force per axis applied to the agent body. */
  maxForce?: number;
  seed?: number;
}

export interface ReachDiagnostics {
  readonly agentPosition: Vec3;
  readonly puckPosition: Vec3;
  readonly goalPosition: Vec3;
  readonly puckDistance: number;
  readonly reached: boolean;
}

export class ReachEnv implements LearningEnvironment {
  readonly name = 'reach';
  readonly observationSize = 14;
  readonly actionSize = 2;
  readonly discount: number;
  readonly maxStepsPerEpisode: number;

  readonly backend: PhysicsBackend;
  private readonly ownsBackend: boolean;
  private readonly fixedDt: number;
  private readonly arenaHalf: number;
  private readonly goalTolerance: number;
  private readonly maxForce: number;
  private readonly rewardWeights: Required<RewardShape>;
  /** Replaced on every `reset()` so callers can drive episode randomisation. */
  private rng: Rng;

  private agent = 0;
  private puck = 0;
  private readonly goal: Vec3[] = [vec3()];
  private readonly observation = new Float32Array(this.observationSize);
  private stepsThisEpisode = 0;
  /** Shaping potential of the previous state, for the Phi(s') - Phi(s) difference. */
  private lastPotential = 0;
  private reached = false;

  constructor(options: ReachEnvOptions = {}) {
    this.fixedDt = options.fixedDt ?? 1 / 60;
    this.ownsBackend = options.backend === undefined;
    this.backend =
      options.backend ??
      new BuiltinPhysics({
        fixedDt: this.fixedDt,
        gravity: vec3(0, 0, 0), // top-down table: no gravity in the play plane
        linearDamping: 2.4,
        angularDamping: 2.4,
        solverIterations: 6,
      });
    this.maxStepsPerEpisode = options.maxStepsPerEpisode ?? 200;
    this.discount = options.discount ?? 0.99;
    this.arenaHalf = options.arenaHalf ?? 1.0;
    // The default regime is tuned to be *learnable* by the bundled REINFORCE
    // trainer in a few thousand episodes while still requiring real contact
    // control: the agent must get behind the puck relative to the goal and push
    // it in, not teleport it. Tightening `goalTolerance` below ~0.15 or cutting
    // `maxForce` makes even a hand-written oracle controller miss most episodes
    // (measured ~3% success at tolerance 0.12), which is a task-design problem,
    // not a trainer problem, so it is relaxed here rather than papered over with
    // reward shaping.
    this.goalTolerance = options.goalTolerance ?? 0.2;
    this.maxForce = options.maxForce ?? 8;
    this.rewardWeights = {
      reach: options.reward?.reach ?? 1,
      success: options.reward?.success ?? 10,
      // Per-step cost of a full-magnitude action. Must stay far below the
      // progress term or the optimal policy becomes "do not move".
      effort: options.reward?.effort ?? 0.002,
      goalTolerance: options.reward?.goalTolerance ?? this.goalTolerance,
      approach: options.reward?.approach ?? 0.5,
      step: options.reward?.step ?? 0.005,
    };
    this.rng = new Rng(options.seed ?? 7);
    this.buildStaticGeometry();
    this.agent = this.backend.createBody({
      shape: { kind: 'sphere', radius: 0.09 },
      position: vec3(0, 0, 0),
      mass: 1,
      friction: 0.9,
      restitution: 0.05,
      label: 'agent',
    });
    this.puck = this.backend.createBody({
      shape: { kind: 'sphere', radius: 0.08 },
      position: vec3(0.4, 0, 0),
      mass: 0.4,
      friction: 0.6,
      restitution: 0.2,
      label: 'puck',
    });
    this.reset(this.rng);
  }

  private buildStaticGeometry(): void {
    const h = this.arenaHalf;
    const t = 0.05; // wall thickness
    const walls: { position: Vec3; halfExtents: Vec3; label: string }[] = [
      { position: vec3(0, 0, -h - t), halfExtents: vec3(h + t, t, t), label: 'wall:-z' },
      { position: vec3(0, 0, h + t), halfExtents: vec3(h + t, t, t), label: 'wall:+z' },
      { position: vec3(-h - t, 0, 0), halfExtents: vec3(t, t, h + t), label: 'wall:-x' },
      { position: vec3(h + t, 0, 0), halfExtents: vec3(t, t, h + t), label: 'wall:+x' },
    ];
    for (const wall of walls) {
      this.backend.createBody({
        shape: { kind: 'box', halfExtents: wall.halfExtents },
        position: wall.position,
        kind: 'static',
        friction: 0.4,
        restitution: 0.3,
        label: wall.label,
      });
    }
  }

  reset(rng: Rng): Float32Array {
    this.rng = rng;
    this.stepsThisEpisode = 0;
    this.reached = false;
    // Rejection-sample the three positions so nothing starts overlapping and
    // every episode requires real control: the agent cannot simply drive
    // forward, it has to get behind the puck relative to the goal.
    const spawn = (): Vec3 =>
      vec3(rng.range(-0.6, 0.6), 0, rng.range(-0.6, 0.6));
    let goal = spawn();
    let puck = spawn();
    let agent = spawn();
    for (let guard = 0; guard < 64; guard++) {
      const ok =
        distanceVec3(goal, puck) >= 0.45 &&
        distanceVec3(agent, puck) >= 0.3 &&
        distanceVec3(agent, goal) >= 0.3;
      if (ok) break;
      if (guard % 3 === 0) goal = spawn();
      else if (guard % 3 === 1) puck = spawn();
      else agent = spawn();
    }
    this.goal[0] = goal;
    this.backend.setBodyState(this.agent, {
      position: agent,
      velocity: vec3(),
      angularVelocity: vec3(),
    });
    this.backend.setBodyState(this.puck, { position: puck, velocity: vec3() });
    this.backend.drainContacts();
    this.lastPotential = this.potential(agent, puck, goal);
    return this.observe(this.observation);
  }

  /**
   * Potential function Phi(s) for shaping: the negated, weighted sum of the
   * puck-goal gap and the agent-puck gap. Larger Phi (closer to 0) is better,
   * so the shaping reward `Phi(s') - Phi(s)` pays for progress along
   * *both* axes — closing on the puck and driving it toward the goal.
   *
   * This is Ng et al. (1999) potential-based reward shaping, which is provably
   * policy-invariant: it leaves the optimal policy unchanged while making the
   * gradient far denser. That matters here because the unshaped task has a flat
   * reward basin — until the agent happens to push the puck into the goal there
   * is no signal that approaching the puck is good, so a policy-gradient method
   * wanders. The earlier hand-tuned `closing * 10` dense term did not help
   * because it rewarded puck-goal motion only, leaving the approach direction
   * (the actually hard part) unsignalled.
   */
  private potential(agentPos: Vec3, puckPos: Vec3, goal: Vec3): number {
    const w = this.rewardWeights;
    const puckGoal = distanceVec3(puckPos, goal);
    const agentPuck = distanceVec3(agentPos, puckPos);
    return -(w.reach * puckGoal + w.approach * agentPuck);
  }

  step(action: ArrayLike<number>): StepResult {
    const ax = clamp(action[0] ?? 0, -1, 1) * this.maxForce;
    const az = clamp(action[1] ?? 0, -1, 1) * this.maxForce;
    this.backend.applyImpulse(this.agent, vec3(ax * this.fixedDt, 0, az * this.fixedDt));
    this.backend.step(this.fixedDt);
    this.backend.drainContacts();
    this.stepsThisEpisode++;

    const agentPos = this.backend.getBodyState(this.agent)?.position ?? vec3();
    const puckState = this.backend.getBodyState(this.puck);
    const puckPos = puckState?.position ?? vec3();
    const goal = this.goal[0]!;
    const distance = distanceVec3(puckPos, goal);
    const potential = this.potential(agentPos, puckPos, goal);
    const shaping = potential - this.lastPotential;
    this.lastPotential = potential;

    const w = this.rewardWeights;
    const tolerance = w.goalTolerance || this.goalTolerance;
    const justReached = !this.reached && distance <= tolerance;
    if (justReached) this.reached = true;

    let reward = -w.step + shaping;
    reward -= w.effort * (Math.abs(ax) + Math.abs(az)) / this.maxForce;
    if (justReached) reward += w.success;

    const done = this.reached || this.stepsThisEpisode >= this.maxStepsPerEpisode;
    return {
      reward,
      done,
      // Reaching the goal ends the task; running out of steps only ends the
      // episode, and the trainer must keep bootstrapping value through it.
      terminated: this.reached,
      info: { distance, shaping, steps: this.stepsThisEpisode },
    };
  }

  observe(out: Float32Array): Float32Array {
    const agent = this.backend.getBodyState(this.agent);
    const puck = this.backend.getBodyState(this.puck);
    const goal = this.goal[0]!;
    const agentPos = agent?.position ?? vec3();
    const puckPos = puck?.position ?? vec3();
    const agentVel = agent?.velocity ?? vec3();
    const puckVel = puck?.velocity ?? vec3();
    const toPuck = subVec3(puckPos, agentPos);
    const toGoal = subVec3(goal, puckPos);
    const dist = lengthVec3(toPuck);
    const goalDist = lengthVec3(toGoal);
    let i = 0;
    // Observations are normalised by arena size so the policy transfers across scales.
    out[i++] = agentPos[0] / this.arenaHalf;
    out[i++] = agentPos[2] / this.arenaHalf;
    out[i++] = puckPos[0] / this.arenaHalf;
    out[i++] = puckPos[2] / this.arenaHalf;
    out[i++] = goal[0] / this.arenaHalf;
    out[i++] = goal[2] / this.arenaHalf;
    out[i++] = dist > 1e-6 ? toPuck[0] / dist : 0;
    out[i++] = dist > 1e-6 ? toPuck[2] / dist : 0;
    out[i++] = goalDist > 1e-6 ? toGoal[0] / goalDist : 0;
    out[i++] = goalDist > 1e-6 ? toGoal[2] / goalDist : 0;
    // Both components of both bodies' planar velocity. Contact tasks are
    // second-order: knowing where the puck is without knowing which way it is
    // already sliding leaves the policy guessing a step behind.
    out[i++] = clamp(agentVel[0] * 0.4, -1, 1);
    out[i++] = clamp(agentVel[2] * 0.4, -1, 1);
    out[i++] = clamp(puckVel[0] * 0.4, -1, 1);
    out[i++] = clamp(puckVel[2] * 0.4, -1, 1);
    return out;
  }

  /** Live state for renderers and tests. */
  diagnostics(): ReachDiagnostics {
    const agent = this.backend.getBodyState(this.agent)?.position ?? vec3();
    const puck = this.backend.getBodyState(this.puck)?.position ?? vec3();
    const goal = this.goal[0]!;
    return {
      agentPosition: agent,
      puckPosition: puck,
      goalPosition: goal,
      puckDistance: distanceVec3(puck, goal),
      reached: this.reached,
    };
  }

  get handles(): { agent: number; puck: number } {
    return { agent: this.agent, puck: this.puck };
  }

  dispose(): void {
    if (this.ownsBackend) this.backend.dispose();
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
