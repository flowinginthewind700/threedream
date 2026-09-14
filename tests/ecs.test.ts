import { describe, expect, it } from 'vitest';

import {
  World,
  defineComponent,
  type System,
  type SystemContext,
} from '../src/core/ecs.js';

interface Transform {
  x: number;
  y: number;
}

interface Velocity {
  vx: number;
  vy: number;
}

const TransformComponent = defineComponent<Transform>('Transform');
const VelocityComponent = defineComponent<Velocity>('Velocity');
const TagComponent = defineComponent<{ tag: string }>('Tag');

function makeWorld(): World {
  return new World();
}

describe('World entities', () => {
  it('issues unique, ascending ids', () => {
    const world = makeWorld();
    const a = world.createEntity();
    const b = world.createEntity();
    expect(a).not.toBe(b);
    expect(b).toBeGreaterThan(a);
    expect(world.entityCount).toBe(2);
  });

  it('tracks liveness and destroys', () => {
    const world = makeWorld();
    const a = world.createEntity();
    expect(world.isAlive(a)).toBe(true);
    expect(world.destroyEntity(a)).toBe(true);
    expect(world.isAlive(a)).toBe(false);
    // Destroying twice is a no-op, not an error.
    expect(world.destroyEntity(a)).toBe(false);
  });

  it('records pending removals until flushed', () => {
    const world = makeWorld();
    const a = world.createEntity();
    const b = world.createEntity();
    world.destroyEntity(a);
    world.destroyEntity(b);
    expect(world.pendingRemovals).toEqual([a, b]);
    world.flush();
    expect(world.pendingRemovals).toEqual([]);
  });

  it('drops all components when an entity is destroyed', () => {
    const world = makeWorld();
    const a = world.createEntity();
    world.add(a, TransformComponent, { x: 1, y: 2 });
    world.destroyEntity(a);
    expect(world.get(a, TransformComponent)).toBeUndefined();
    expect(world.query(TransformComponent)).toEqual([]);
  });

  it('entities() returns a sorted copy', () => {
    const world = makeWorld();
    const ids = [world.createEntity(), world.createEntity(), world.createEntity()];
    const listed = world.entities();
    expect(listed).toEqual([...ids].sort((a, b) => a - b));
    listed.length = 0;
    expect(world.entityCount).toBe(3);
  });

  it('clear() resets ids, counters and stores', () => {
    const world = makeWorld();
    const a = world.createEntity();
    world.add(a, TransformComponent, { x: 0, y: 0 });
    world.step(1 / 60);
    world.clear();
    expect(world.entityCount).toBe(0);
    expect(world.stepCounter).toBe(0);
    expect(world.simulatedSeconds).toBe(0);
    expect(world.createEntity()).toBe(1);
  });
});

describe('World components', () => {
  it('stores, reads back and reports presence', () => {
    const world = makeWorld();
    const a = world.createEntity();
    world.add(a, TransformComponent, { x: 3, y: -1 });
    expect(world.has(a, TransformComponent)).toBe(true);
    expect(world.get(a, TransformComponent)).toEqual({ x: 3, y: -1 });
    expect(world.require(a, TransformComponent).x).toBe(3);
  });

  it('keeps component stores separate', () => {
    const world = makeWorld();
    const a = world.createEntity();
    world.add(a, TransformComponent, { x: 1, y: 1 });
    expect(world.has(a, VelocityComponent)).toBe(false);
    expect(world.get(a, VelocityComponent)).toBeUndefined();
  });

  it('overwrites on a second add', () => {
    const world = makeWorld();
    const a = world.createEntity();
    world.add(a, TransformComponent, { x: 1, y: 1 });
    world.add(a, TransformComponent, { x: 9, y: 9 });
    expect(world.get(a, TransformComponent)).toEqual({ x: 9, y: 9 });
  });

  it('require() throws for a missing component', () => {
    const world = makeWorld();
    const a = world.createEntity();
    expect(() => world.require(a, TransformComponent)).toThrow(/Transform/);
  });

  it('add() refuses a dead entity', () => {
    const world = makeWorld();
    const a = world.createEntity();
    world.destroyEntity(a);
    expect(() => world.add(a, TransformComponent, { x: 0, y: 0 })).toThrow(/not alive/);
  });

  it('remove() reports whether anything was there', () => {
    const world = makeWorld();
    const a = world.createEntity();
    world.add(a, TransformComponent, { x: 0, y: 0 });
    expect(world.remove(a, TransformComponent)).toBe(true);
    expect(world.remove(a, TransformComponent)).toBe(false);
  });
});

describe('World queries', () => {
  function fixture(): World {
    const world = makeWorld();
    const both = world.createEntity();
    world.add(both, TransformComponent, { x: 0, y: 0 });
    world.add(both, VelocityComponent, { vx: 1, vy: 1 });
    const transformOnly = world.createEntity();
    world.add(transformOnly, TransformComponent, { x: 1, y: 1 });
    const tagged = world.createEntity();
    world.add(tagged, TransformComponent, { x: 2, y: 2 });
    world.add(tagged, TagComponent, { tag: 'x' });
    world.add(tagged, VelocityComponent, { vx: 0, vy: 0 });
    return world;
  }

  it('intersects component requirements', () => {
    const world = fixture();
    expect(world.query(TransformComponent)).toHaveLength(3);
    expect(world.query(TransformComponent, VelocityComponent)).toHaveLength(2);
    expect(world.query(TransformComponent, VelocityComponent, TagComponent)).toHaveLength(1);
  });

  it('returns empty when a store does not exist yet', () => {
    const world = makeWorld();
    world.createEntity();
    expect(world.query(defineComponent<never>('NeverAdded'))).toEqual([]);
  });

  it('with no tokens lists every live entity', () => {
    const world = fixture();
    expect(world.query()).toEqual(world.entities());
  });

  it('excludes destroyed entities even if their components linger in a store', () => {
    const world = fixture();
    const ids = world.query(TransformComponent, VelocityComponent);
    world.destroyEntity(ids[1]!);
    expect(world.query(TransformComponent, VelocityComponent)).not.toContain(ids[1]!);
  });

  it('results are ascending by id', () => {
    const world = fixture();
    const ids = world.query(TransformComponent);
    expect(ids).toEqual([...ids].sort((a, b) => a - b));
  });
});

describe('World systems', () => {
  function recorder(log: string[], name: string, order?: number, phase?: System['phase']): System {
    return {
      name,
      order,
      phase,
      update: () => {
        log.push(name);
      },
    };
  }

  it('runs systems in phase order, then by `order`', () => {
    const world = makeWorld();
    const log: string[] = [];
    world.addSystem(recorder(log, 'update-late', 10));
    world.addSystem(recorder(log, 'post', 0, 'post-update'));
    world.addSystem(recorder(log, 'update-early', -10));
    world.addSystem(recorder(log, 'pre', 0, 'pre-update'));
    world.step(1 / 60);
    expect(log).toEqual(['pre', 'update-early', 'update-late', 'post']);
  });

  it('passes a monotonic step counter and accumulating time', () => {
    const world = makeWorld();
    const seen: Array<[number, number]> = [];
    world.addSystem({
      name: 'probe',
      update: (ctx: SystemContext) => {
        seen.push([ctx.step, ctx.time]);
      },
    });
    world.step(0.5);
    world.step(0.5);
    world.step(0.5);
    expect(seen.map(([s]) => s)).toEqual([0, 1, 2]);
    expect(seen.map(([, t]) => t)).toEqual([0, 0.5, 1.0]);
    expect(world.stepCounter).toBe(3);
    expect(world.simulatedSeconds).toBeCloseTo(1.5, 10);
  });

  it('systems can mutate components', () => {
    const world = makeWorld();
    const a = world.createEntity();
    world.add(a, TransformComponent, { x: 0, y: 0 });
    world.add(a, VelocityComponent, { vx: 2, vy: 0 });
    world.addSystem({
      name: 'integrate',
      update: (ctx) => {
        for (const id of ctx.world.query(TransformComponent, VelocityComponent)) {
          const t = ctx.world.require(id, TransformComponent);
          const v = ctx.world.require(id, VelocityComponent);
          t.x += v.vx * ctx.dt;
          t.y += v.vy * ctx.dt;
        }
      },
    });
    world.step(1);
    world.step(1);
    expect(world.get(a, TransformComponent)).toEqual({ x: 4, y: 0 });
  });

  it('removeSystem drops it by name and reports the result', () => {
    const world = makeWorld();
    const log: string[] = [];
    world.addSystem(recorder(log, 'a'));
    world.addSystem(recorder(log, 'b'));
    expect(world.removeSystem('a')).toBe(true);
    expect(world.removeSystem('missing')).toBe(false);
    world.step(1 / 60);
    expect(log).toEqual(['b']);
    expect(world.systemNames).toEqual(['b']);
  });
});
