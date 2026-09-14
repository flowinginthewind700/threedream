/**
 * CI/CD contract tests.
 *
 * Workflows are config, not code: nothing type-checks them, and a typo fails
 * silently as a skipped run rather than as a red build. These tests parse the
 * YAML and assert the parts that matter, so a regression shows up as a local
 * test failure instead of a mysteriously un-run pipeline.
 *
 * Parsed with `yaml` rather than regex-matched, because an indented `run:`
 * string is easy to match by accident while asserting something the runner
 * never actually does.
 */

import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';

import pkg from '../package.json';

const ROOT = resolve(import.meta.dirname, '..');

interface WorkflowStep {
  name?: string;
  if?: string;
  uses?: string;
  run?: string;
  with?: Record<string, unknown>;
}

interface WorkflowJob {
  name?: string;
  'runs-on'?: string;
  needs?: string | string[];
  if?: string;
  permissions?: Record<string, string>;
  environment?: string | { name: string; url?: string };
  steps?: WorkflowStep[];
}

interface Workflow {
  name?: string;
  /** GitHub Actions writes the key `on:`; YAML keeps it as the string 'on'. */
  on?: Record<string, unknown> | string | unknown[];
  permissions?: Record<string, string>;
  concurrency?: { group?: string; 'cancel-in-progress'?: boolean };
  jobs?: Record<string, WorkflowJob>;
}

/** Gate jobs: everything a push must pass before anything can be published. */
const GATES = ['verify', 'coverage', 'e2e'] as const;

function loadWorkflow(file: string): Workflow {
  const text = readFileSync(resolve(ROOT, '.github/workflows', file), 'utf8');
  return parse(text) as Workflow;
}

const ci = loadWorkflow('ci.yml');
const jobs = ci.jobs ?? {};

function runs(job: WorkflowJob): string {
  return (job.steps ?? []).map((s) => s.run ?? '').join('\n');
}

function usesList(job: WorkflowJob): string[] {
  return (job.steps ?? []).map((s) => s.uses ?? '').filter(Boolean);
}

function stepByUses(job: WorkflowJob, prefix: string): WorkflowStep | undefined {
  return (job.steps ?? []).find((s) => s.uses?.startsWith(prefix));
}

function environmentName(job: WorkflowJob): string {
  const e = job.environment;
  return typeof e === 'string' ? e : (e?.name ?? '');
}

function pushBranches(w: Workflow): string[] {
  const push = (w.on ?? {}) as Record<string, unknown>;
  const value = push.push as { branches?: string[] } | undefined;
  return value?.branches ?? [];
}

describe('ci.yml — triggers and shape', () => {
  it('is named and runs on pushes to main, pull requests, and manual dispatch', () => {
    expect(ci.name, 'workflow needs a name').toBeTruthy();
    const on = ci.on as Record<string, unknown>;
    expect(on.push, 'missing push trigger').toBeDefined();
    expect(on.pull_request, 'missing pull_request trigger').toBeDefined();
    expect(on.workflow_dispatch, 'manual dispatch is required to re-run a deploy').toBeDefined();
    expect(pushBranches(ci)).toContain('main');
  });

  it('has exactly the four jobs: three gates and one deploy', () => {
    // One file, one graph. GitHub Actions cannot express `needs:` across
    // workflow files, so a separate pages.yml would deploy main even when the
    // gate is red -- the one failure mode CI exists to prevent.
    expect(Object.keys(jobs).sort()).toEqual([...GATES, 'deploy'].sort());
  });

  it('every job declares a GitHub-hosted Ubuntu runner and has steps', () => {
    for (const [id, job] of Object.entries(jobs)) {
      expect(job['runs-on'], `${id} needs runs-on`).toMatch(/^ubuntu-/);
      expect((job.steps ?? []).length, `${id} needs steps`).toBeGreaterThan(0);
    }
  });

  it('cancels a stale run when a newer push arrives', () => {
    expect(ci.concurrency?.group).toMatch(/\$\{\{/);
    expect(ci.concurrency?.['cancel-in-progress']).toBe(true);
  });
});

describe('every job installs identically', () => {
  it.each(Object.entries(jobs))('%s checks out, pins Node, and installs from the lockfile', (_id, job) => {
    expect(stepByUses(job, 'actions/checkout'), 'actions/checkout is required').toBeDefined();

    const setup = stepByUses(job, 'actions/setup-node');
    expect(setup, 'actions/setup-node is required').toBeDefined();
    // CI proves the engine floor package.json advertises, so the two cannot drift.
    const floor = String(pkg.engines?.node ?? '').replace(/[^\d.]/g, '').split('.')[0];
    expect(floor, 'package.json must declare an engines floor').toBeTruthy();
    expect(String(setup?.with?.['node-version']), 'node-version must honour the engines floor').toContain(floor);
    // Lockfile-based caching keeps the pipeline fast and reproducible.
    expect(setup?.with?.cache).toBe('npm');

    // `npm ci` is what makes the CI dependency tree match a local one exactly.
    expect(runs(job)).toMatch(/npm ci\b/);
  });

  it('no job recurses into the Unreal Engine submodule', () => {
    // thirdparty/UnrealEngine is a gitlink to Epic's private repo and runners
    // have no EULA access, so a recursive checkout would fail every single run.
    for (const [id, job] of Object.entries(jobs)) {
      const checkout = stepByUses(job, 'actions/checkout');
      expect(
        String(checkout?.with?.submodules ?? 'false'),
        `${id}: submodule checkout must not recurse into Epic's private repo`,
      ).not.toMatch(/recursive|true/);
      expect(runs(job), `${id}: no manual submodule update`).not.toMatch(/submodule update/);
    }
  });

  it('no step swallows a failing command', () => {
    // `|| true` would let a broken gate report success, which is worse than no
    // gate at all. Every command must be able to fail the run.
    for (const [id, job] of Object.entries(jobs)) {
      for (const step of job.steps ?? []) {
        if (step.run) expect(step.run, `${id}: a step must not swallow failures`).not.toMatch(/\|\|\s*true/);
      }
    }
  });
});

describe('the three gates each enforce one thing', () => {
  it('verify gates on typecheck, the full unit suite, and the production build', () => {
    const run = runs(jobs.verify!);
    expect(run).toMatch(/npm run typecheck/);
    expect(run).toMatch(/npm test/);
    expect(run).toMatch(/npm run build/);
  });

  it('coverage runs the threshold gate in parallel, not inside the fast loop', () => {
    // Coverage is a threshold gate, not a correctness gate: `verify` already
    // runs the same tests un-instrumented. Keeping it in its own job means a
    // coverage regression cannot mask a real test failure behind a slow runner,
    // and the two finish at about the same time instead of in series.
    expect(runs(jobs.coverage!)).toMatch(/npm run test:coverage/);
    expect(jobs.coverage!.needs, 'coverage must not wait on verify').toBeUndefined();
    expect(runs(jobs.verify!), 'verify must not run coverage').not.toMatch(/--coverage|test:coverage/);
  });

  it('e2e covers the render layer the unit suite cannot reach', () => {
    const run = runs(jobs.e2e!);
    expect(run).toMatch(/npm run test:e2e/);
    // A fresh runner has no browsers cached; without this the job fails on
    // "Executable doesn't exist" rather than on an actual assertion.
    expect(run, 'browsers must be installed on the runner').toMatch(/playwright install/);
    expect(run).toMatch(/chromium/);
    expect(jobs.e2e!.needs, 'e2e must not wait on verify').toBeUndefined();
  });

  it('gate jobs cannot publish: no pages/id-token permissions, no deploy actions', () => {
    for (const id of GATES) {
      const job = jobs[id]!;
      expect(job.permissions?.pages, `${id} must not get pages:write`).toBeUndefined();
      expect(job.permissions?.['id-token'], `${id} must not get id-token:write`).toBeUndefined();
      expect(usesList(job).join(' '), `${id} must not deploy`).not.toMatch(/upload-pages-artifact|deploy-pages/);
    }
    // Workflow-level default stays read-only; only `deploy` widens it.
    expect(ci.permissions ?? {}).toMatchObject({ contents: 'read' });
    expect(ci.permissions ?? {}).not.toHaveProperty('pages');
    expect(ci.permissions ?? {}).not.toHaveProperty('id-token');
  });

  it('every action in every job is pinned to a full commit SHA, not a mutable tag', () => {
    // A tag like actions/checkout@v4 can be force-pushed upstream, silently
    // changing what CI executes. A 40-hex SHA cannot.
    for (const [id, job] of Object.entries(jobs)) {
      for (const uses of usesList(job)) {
        expect(uses, `${id}: unpinned action ${uses}`).toMatch(/@[0-9a-f]{40}$/);
      }
      expect(usesList(job).length, `${id}: expected at least one action`).toBeGreaterThan(0);
    }
  });
});

describe('deploy — continuous deployment of the demo', () => {
  const deploy = jobs.deploy!;

  it('needs every gate, so red tests cannot publish', () => {
    const needs = Array.isArray(deploy.needs) ? deploy.needs : [deploy.needs];
    expect(needs.sort(), 'deploy must be gated by all three jobs').toEqual([...GATES].sort());
    // And only on main: a PR run must build the artifact, never ship it.
    expect(deploy.if, 'deploy must be restricted to pushes on main').toMatch(/refs\/heads\/main/);
    expect(deploy.if).toMatch(/github\.event_name == 'push'/);
  });

  it('grants exactly the permissions Pages deployment needs, on the job only', () => {
    expect(deploy.permissions).toMatchObject({
      contents: 'read',
      pages: 'write',
      'id-token': 'write',
    });
    expect(environmentName(deploy)).toBe('github-pages');
  });

  it('builds with the Pages base path injected', () => {
    // The site is served from /<repo>/, so the build has to know its own prefix.
    // vite.config.ts reads VITE_BASE and defaults to '/'; `build:pages` sets it
    // to /<repo>/, which is what makes the emitted asset refs correct.
    expect(runs(deploy)).toMatch(/build:pages/);
    const pkgJson = JSON.parse(readFileSync(resolve(ROOT, 'package.json'), 'utf8')) as {
      scripts: Record<string, string>;
    };
    expect(pkgJson.scripts['build:pages'], 'build:pages must inject VITE_BASE').toMatch(/VITE_BASE/);
    expect(pkgJson.scripts['build:pages']).toMatch(/npm run build/);
    expect(stepByUses(deploy, 'actions/setup-node'), 'deploy build needs node').toBeDefined();
  });

  it('uploads dist/ as the Pages artifact', () => {
    const upload = stepByUses(deploy, 'actions/upload-pages-artifact');
    expect(upload, 'upload-pages-artifact step is required').toBeDefined();
    expect(upload?.with?.path).toBe('dist');
  });

  it('deploys the uploaded artifact via OIDC', () => {
    const steps = deploy.steps ?? [];
    const uploadIdx = steps.findIndex((s) => s.uses?.startsWith('actions/upload-pages-artifact'));
    const deployIdx = steps.findIndex((s) => s.uses?.startsWith('actions/deploy-pages'));
    expect(uploadIdx, 'upload step required').toBeGreaterThanOrEqual(0);
    expect(deployIdx, 'deploy step required').toBeGreaterThan(uploadIdx);
  });
});

describe('there is no second workflow that could deploy around the gate', () => {
  it('.github/workflows holds only ci.yml', () => {
    const files = readdirSync(resolve(ROOT, '.github/workflows')).filter((f) => /\.(ya?ml)$/.test(f));
    expect(files, 'a stray workflow would bypass the `needs:` graph').toEqual(['ci.yml']);
  });
});
