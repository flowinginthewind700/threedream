/**
 * Headless training run. No browser, no WebGL, no WASM.
 *
 *   npm run train                      # default budget
 *   npm run train -- --episodes 4000 --seed 3 --out artifacts/policy.json
 *
 * Trains a Gaussian policy on the `reach` task with the built-in deterministic
 * solver and prints a progress trace. The same loop drives the browser demo, so
 * a policy that trains here is a policy that plays there.
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

import { GaussianPolicy } from '../src/ai/policy.js';
import { Trainer } from '../src/ai/trainer.js';
import { ReachEnv } from '../src/envs/reach.js';
import { Rng } from '../src/core/rng.js';

interface CliOptions {
  episodes: number;
  seed: number;
  out?: string;
  every: number;
  hidden: number[];
  learningRate: number;
  batch: number;
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    episodes: 800,
    seed: 20260914,
    every: 100,
    hidden: [64, 64],
    learningRate: 0.02,
    batch: 16,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      const value = argv[++i];
      if (value === undefined) throw new Error(`missing value for ${arg}`);
      return value;
    };
    switch (arg) {
      case '--episodes':
        options.episodes = Number(next());
        break;
      case '--seed':
        options.seed = Number(next());
        break;
      case '--out':
        options.out = next();
        break;
      case '--every':
        options.every = Number(next());
        break;
      case '--lr':
        options.learningRate = Number(next());
        break;
      case '--batch':
        options.batch = Number(next());
        break;
      case '--hidden':
        options.hidden = next()
          .split(',')
          .map((n) => Number(n))
          .filter((n) => Number.isFinite(n) && n > 0);
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  for (const [key, value] of Object.entries(options)) {
    if (typeof value === 'number' && !Number.isFinite(value)) {
      throw new Error(`--${key} must be a finite number`);
    }
  }
  return options;
}

function printHelp(): void {
  console.log(`threedream headless trainer

  --episodes N   episodes to train (default 800)
  --seed N       RNG seed for env + policy init (default 20260914)
  --lr N         policy learning rate (default 0.02)
  --batch N      episodes per policy update (default 16)
  --hidden a,b   MLP hidden layer sizes (default 64,64)
  --every N      log every N episodes (default 100)
  --out PATH     write the trained policy JSON here`);
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const startedAt = Date.now();

  const env = new ReachEnv({ seed: options.seed });
  const policy = new GaussianPolicy({
    observationSize: env.observationSize,
    actionSize: env.actionSize,
    hiddenSizes: options.hidden,
    seed: options.seed,
    actionScale: 1,
  });
  const trainer = new Trainer({
    learningRate: options.learningRate,
    entropyCoefficient: 0.002,
    episodesPerUpdate: options.batch,
    seed: options.seed,
  });

  console.log(
    `threedream train: task=${env.name} backend=${env.backend.name} ` +
      `obs=${env.observationSize} act=${env.actionSize} ` +
      `params=${policy.parameterCount} episodes=${options.episodes} ` +
      `batch=${options.batch} seed=${options.seed}`,
  );

  let best = Number.NEGATIVE_INFINITY;
  let bestEpisode = -1;

  const result = trainer.train(policy, env, options.episodes, (episode) => {
    if (episode.return > best) {
      best = episode.return;
      bestEpisode = episode.episode;
    }
    if (
      episode.episode % options.every === 0 ||
      episode.episode === options.episodes - 1
    ) {
      console.log(
        `  ep ${String(episode.episode).padStart(5)} ` +
          `return ${episode.return.toFixed(3).padStart(9)} ` +
          `batchRet ${episode.batchMeanReturn.toFixed(3).padStart(9)} ` +
          `steps ${String(episode.steps).padStart(4)} ` +
          `entropy ${episode.entropy.toFixed(3)} ` +
          `vloss ${episode.valueLoss.toFixed(4)} ` +
          `advStd ${episode.advantageStd.toFixed(4)}`,
      );
    }
  });

  // Greedy evaluation: the deployment-mode metric, distinct from training return.
  const evaluation = evaluateGreedy(env, policy, 20, options.seed + 1);

  const elapsed = (Date.now() - startedAt) / 1000;
  console.log('');
  console.log(
    `done in ${elapsed.toFixed(1)}s  steps=${result.steps}  ` +
      `bestTrainReturn=${best.toFixed(3)}@ep${bestEpisode}  ` +
      `greedySuccess=${(evaluation.successRate * 100).toFixed(0)}%  ` +
      `greedyMeanReturn=${evaluation.meanReturn.toFixed(3)}`,
  );

  if (options.out) {
    const path = resolve(process.cwd(), options.out);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(
      path,
      JSON.stringify(
        {
          task: env.name,
          backend: env.backend.name,
          seed: options.seed,
          episodes: options.episodes,
          trainedAt: new Date().toISOString(),
          greedySuccessRate: evaluation.successRate,
          greedyMeanReturn: evaluation.meanReturn,
          policy: policy.toJSON(),
        },
        null,
        2,
      ),
    );
    console.log(`policy written to ${path}`);
  }

  env.dispose();
}

interface EvaluationResult {
  successRate: number;
  meanReturn: number;
  meanSteps: number;
}

/** Run the policy greedily (mean action) and report success rate. */
function evaluateGreedy(
  env: ReachEnv,
  policy: GaussianPolicy,
  episodes: number,
  seed: number,
): EvaluationResult {
  const rng = new Rng(seed);
  const action = new Float32Array(policy.actionSize);
  const observation = new Float32Array(policy.observationSize);
  let successes = 0;
  let returnSum = 0;
  let stepSum = 0;

  for (let e = 0; e < episodes; e++) {
    env.reset(rng);
    let total = 0;
    let steps = 0;
    let done = false;
    while (!done && steps < env.maxStepsPerEpisode) {
      policy.actGreedy(env.observe(observation), action);
      const result = env.step(action);
      total += result.reward;
      done = result.done;
      steps++;
    }
    if (env.diagnostics().reached) successes++;
    returnSum += total;
    stepSum += steps;
  }

  return {
    successRate: successes / Math.max(1, episodes),
    meanReturn: returnSum / Math.max(1, episodes),
    meanSteps: stepSum / Math.max(1, episodes),
  };
}

main();
