import { describe, expect, it } from 'vitest';

import { Rng } from '../src/core/rng.js';
import { DriveEnv } from '../src/envs/drive.js';
import { ReachEnv } from '../src/envs/reach.js';
import { distanceVec3 } from '../src/physics/types.js';

/** Drive a scripted action for one episode and total the reward. */
function rollout(
  env: DriveEnv | ReachEnv,
  rng: Rng,
  controller: (observation: Float32Array, action: Float32Array, step: number) => void,
): { reward: number; steps: number; reached: boolean } {
  const action = new Float32Array(env.actionSize);
  const buffer = new Float32Array(env.observationSize);
  env.reset(rng);
  let total = 0;
  let steps = 0;
  let done = false;
  while (!done && steps < env.maxStepsPerEpisode) {
    const observation = env.observe(buffer);
    controller(observation, action, steps);
    const result = env.step(action);
    total += result.reward;
    done = result.done;
    steps++;
  }
  return { reward: total, steps, reached: env.diagnostics().reached };
}

describe('DriveEnv', () => {
  it('declares matching sizes and owns a backend', () => {
    const env = new DriveEnv();
    expect(env.observationSize).toBe(6);
    expect(env.actionSize).toBe(2);
    expect(env.backend.bodyCount).toBe(1);
    expect(env.backend.name).toBe('builtin');
    env.dispose();
  });

  it('reset() returns a fresh observation array of the declared size', () => {
    const env = new DriveEnv();
    const obs = env.reset(new Rng(3));
    expect(obs).toHaveLength(env.observationSize);
    for (const value of obs) expect(Number.isFinite(value)).toBe(true);
    // observe() writes into the caller buffer and returns it.
    const buffer = new Float32Array(env.observationSize);
    expect(env.observe(buffer)).toBe(buffer);
    env.dispose();
  });

  it('separates randomised episodes', () => {
    const env = new DriveEnv();
    const a = Array.from(env.reset(new Rng(1)));
    const b = Array.from(env.reset(new Rng(2)));
    expect(a).not.toEqual(b);
    // The same seed replays the same start.
    const c = Array.from(env.reset(new Rng(1)));
    expect(c).toEqual(a);
    env.dispose();
  });

  it('rewards acting over idling', () => {
    const env = new DriveEnv({ seed: 1 });
    const idle = rollout(env, new Rng(21), (_o, a) => {
      a[0] = 0;
      a[1] = 0;
    });
    // Head straight for the goal using the unit direction in the observation.
    const seeking = rollout(env, new Rng(21), (o, a) => {
      a[0] = o[2]!;
      a[1] = o[3]!;
    });
    expect(idle.reached).toBe(false);
    expect(seeking.reached).toBe(true);
    expect(seeking.reward).toBeGreaterThan(idle.reward);
    env.dispose();
  });

  it('terminates on goal and reports terminated=true', () => {
    const env = new DriveEnv({ seed: 2 });
    env.reset(new Rng(4));
    let terminated = false;
    for (let i = 0; i < env.maxStepsPerEpisode; i++) {
      const obs = env.observe(new Float32Array(env.observationSize));
      const result = env.step([obs[2]!, obs[3]!]);
      if (result.done) {
        terminated = result.terminated ?? false;
        break;
      }
    }
    expect(terminated).toBe(true);
    expect(env.diagnostics().distance).toBeLessThanOrEqual(env.goalTolerance);
    env.dispose();
  });

  it('truncates at the step budget without claiming termination', () => {
    const env = new DriveEnv({ seed: 3 });
    env.reset(new Rng(5));
    let last: StepResultLike | undefined;
    for (let i = 0; i < env.maxStepsPerEpisode; i++) last = env.step([0, 0]);
    expect(last!.done).toBe(true);
    expect(last!.terminated).toBe(false);
    expect(env.diagnostics().reached).toBe(false);
    env.dispose();
  });

  it('potential shaping telescopes: oscillating cannot farm reward', () => {
    const env = new DriveEnv({ seed: 6, reward: { success: 0 } });
    const rng = new Rng(31);
    env.reset(rng);
    // Alternate two opposite pushes from a standstill; net displacement ~0.
    let total = 0;
    for (let i = 0; i < 60; i++) total += env.step([i % 2 === 0 ? 1 : -1, 0]).reward;
    const diagnostics = env.diagnostics();
    // Reward stays bounded by the step cost plus the (unchanged) potential.
    expect(Math.abs(total)).toBeLessThan(0.5 + 60 * 0.006);
    expect(Number.isFinite(diagnostics.distance)).toBe(true);
    env.dispose();
  });

  it('clamps out-of-range actions', () => {
    const env = new DriveEnv({ seed: 8 });
    env.reset(new Rng(9));
    const wild = env.step([1000, -1000]);
    env.reset(new Rng(9));
    const saturated = env.step([1, -1]);
    expect(wild.reward).toBeCloseTo(saturated.reward, 6);
    env.dispose();
  });
});

interface StepResultLike {
  reward: number;
  done: boolean;
  terminated?: boolean;
}

describe('ReachEnv', () => {
  it('declares a 14-dim observation and 2-dim action', () => {
    const env = new ReachEnv();
    expect(env.observationSize).toBe(14);
    expect(env.actionSize).toBe(2);
    // agent + puck + 4 walls
    expect(env.backend.bodyCount).toBe(6);
    env.dispose();
  });

  it('reset() honours spawn separation constraints', () => {
    const env = new ReachEnv();
    const rng = new Rng(12);
    for (let episode = 0; episode < 50; episode++) {
      env.reset(rng);
      const d = env.diagnostics();
      expect(distanceVec3(d.goalPosition, d.puckPosition)).toBeGreaterThanOrEqual(0.44);
      expect(distanceVec3(d.agentPosition, d.puckPosition)).toBeGreaterThanOrEqual(0.29);
      expect(d.reached).toBe(false);
    }
    env.dispose();
  });

  it('keeps the puck inside the arena under hard random play', () => {
    const env = new ReachEnv({ seed: 15 });
    const rng = new Rng(16);
    const actionRng = new Rng(17);
    const limit = 1.0 + 0.09; // arena half + puck radius, plus solver slop
    for (let episode = 0; episode < 8; episode++) {
      env.reset(rng);
      for (let i = 0; i < env.maxStepsPerEpisode; i++) {
        env.step([actionRng.range(-3, 3), actionRng.range(-3, 3)]);
        const puck = env.diagnostics().puckPosition;
        expect(Math.abs(puck[0])).toBeLessThan(limit);
        expect(Math.abs(puck[2])).toBeLessThan(limit);
      }
    }
    env.dispose();
  });

  it('idling costs exactly the step penalty and earns no shaping', () => {
    const env = new ReachEnv({ seed: 20 });
    const result = rollout(env, new Rng(40), (_o, a) => {
      a[0] = 0;
      a[1] = 0;
    });
    // Phi never changes when nothing moves, so the shaping term must be exactly
    // zero and the episode return must be the flat living cost. This is the
    // invariant that stops "do nothing" from becoming a reward-farming policy.
    expect(result.steps).toBe(env.maxStepsPerEpisode);
    expect(result.reward).toBeCloseTo(-0.005 * result.steps, 6);
    expect(result.reached).toBe(false);
    env.dispose();
  });

  it('shaping telescopes to the potential difference over the episode', () => {
    const env = new ReachEnv({ seed: 24, reward: { success: 0 } });
    const rng = new Rng(43);
    const actionRng = new Rng(44);
    env.reset(rng);
    const start = env.diagnostics();
    // Phi(s) = -(reach * puckGoal + approach * agentPuck), defaults 1 and 0.5.
    const phi = (d: { puckDistance: number; agentPosition: readonly number[]; puckPosition: readonly number[] }): number =>
      -(d.puckDistance + 0.5 * distanceVec3(d.agentPosition as never, d.puckPosition as never));
    const phi0 = phi(start);
    let shapingTotal = 0;
    for (let i = 0; i < env.maxStepsPerEpisode; i++) {
      const result = env.step([actionRng.range(-1, 1), actionRng.range(-1, 1)]);
      shapingTotal += result.info?.shaping ?? 0;
      if (result.done) break;
    }
    const phiEnd = phi(env.diagnostics());
    expect(shapingTotal).toBeCloseTo(phiEnd - phi0, 5);
    env.dispose();
  });

  it('reports distance and reached through diagnostics', () => {
    const env = new ReachEnv({ seed: 22, goalTolerance: 0.2 });
    env.reset(new Rng(41));
    const before = env.diagnostics();
    expect(before.puckDistance).toBeGreaterThan(before.reached ? 0 : 0);
    expect(before.reached).toBe(false);
    // Force the puck onto the goal and confirm the env declares success.
    const { puck } = env.handles;
    env.backend.setBodyState(puck, {
      position: env.diagnostics().goalPosition,
      velocity: [0, 0, 0],
    });
    const result = env.step([0, 0]);
    expect(result.terminated).toBe(true);
    expect(result.done).toBe(true);
    expect(env.diagnostics().reached).toBe(true);
    env.dispose();
  });

  it('accepts an injected backend without disposing it', () => {
    const env = new ReachEnv();
    const shared = new ReachEnv({ backend: env.backend });
    expect(shared.backend).toBe(env.backend);
    shared.dispose();
    expect(env.backend.bodyCount).toBeGreaterThan(0);
    env.dispose();
    expect(env.backend.bodyCount).toBe(0);
  });
});
