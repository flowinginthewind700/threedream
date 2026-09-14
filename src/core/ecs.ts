/**
 * Entity-Component-System core.
 *
 * A game engine's job is to run a lot of small behaviours over a lot of
 * entities without them knowing about each other. Entities are integer ids,
 * components are plain data stored in typed maps, systems are functions run in
 * a declared order against a fixed timestep.
 *
 * Deliberately simple on purpose: this is a skeleton meant to be replaced by a
 * real archetype/bitset ECS later without changing call sites.
 */

export type EntityId = number;

/** Nominal tag for a component store, created by `defineComponent`. */
export interface ComponentToken<T> {
  readonly componentName: string;
  readonly __type?: T;
}

export function defineComponent<T>(name: string): ComponentToken<T> {
  return { componentName: name };
}

export interface SystemContext {
  readonly world: World;
  /** Fixed simulation delta, seconds. */
  readonly dt: number;
  /** Total simulated time, seconds. */
  readonly time: number;
  /** Monotonic step counter. */
  readonly step: number;
}

export interface System {
  readonly name: string;
  /** Lower runs earlier within a phase. */
  readonly order?: number;
  readonly phase?: SystemPhase;
  update(ctx: SystemContext): void;
}

export type SystemPhase = 'pre-update' | 'update' | 'post-update';

const PHASES: SystemPhase[] = ['pre-update', 'update', 'post-update'];

export class World {
  private nextId: EntityId = 1;
  private readonly alive = new Set<EntityId>();
  private readonly removed: EntityId[] = [];
  private readonly stores = new Map<string, Map<EntityId, unknown>>();
  private readonly systems: System[] = [];
  private systemsDirty = true;
  private stepCount = 0;
  private simulatedTime = 0;

  createEntity(): EntityId {
    const id = this.nextId++;
    this.alive.add(id);
    return id;
  }

  destroyEntity(id: EntityId): boolean {
    if (!this.alive.delete(id)) return false;
    for (const store of this.stores.values()) store.delete(id);
    this.removed.push(id);
    return true;
  }

  /** Entities destroyed since the last `flush()`. */
  get pendingRemovals(): readonly EntityId[] {
    return this.removed;
  }

  flush(): void {
    this.removed.length = 0;
  }

  isAlive(id: EntityId): boolean {
    return this.alive.has(id);
  }

  get entityCount(): number {
    return this.alive.size;
  }

  /** All live entity ids, ascending. Copy — do not mutate. */
  entities(): EntityId[] {
    return [...this.alive].sort((a, b) => a - b);
  }

  add<T>(id: EntityId, token: ComponentToken<T>, component: T): T {
    if (!this.alive.has(id)) throw new Error(`entity ${id} is not alive`);
    const store = this.storeFor(token.componentName);
    store.set(id, component);
    return component;
  }

  get<T>(id: EntityId, token: ComponentToken<T>): T | undefined {
    return this.stores.get(token.componentName)?.get(id) as T | undefined;
  }

  require<T>(id: EntityId, token: ComponentToken<T>): T {
    const found = this.get(id, token);
    if (found === undefined) {
      throw new Error(`entity ${id} has no component "${token.componentName}"`);
    }
    return found;
  }

  has<T>(id: EntityId, token: ComponentToken<T>): boolean {
    return this.stores.get(token.componentName)?.has(id) === true;
  }

  remove<T>(id: EntityId, token: ComponentToken<T>): boolean {
    return this.stores.get(token.componentName)?.delete(id) === true;
  }

  /** Every live entity carrying all of `tokens`, ascending by id. */
  query(...tokens: ComponentToken<unknown>[]): EntityId[] {
    if (tokens.length === 0) return this.entities();
    const names = tokens.map((t) => t.componentName);
    // Iterate the rarest store to keep the common case cheap.
    let rarest: Map<EntityId, unknown> | undefined;
    for (const name of names) {
      const store = this.stores.get(name);
      if (!store) return [];
      if (!rarest || store.size < rarest.size) rarest = store;
    }
    const out: EntityId[] = [];
    for (const id of rarest!.keys()) {
      if (!this.alive.has(id)) continue;
      let ok = true;
      for (const name of names) {
        if (!this.stores.get(name)?.has(id)) {
          ok = false;
          break;
        }
      }
      if (ok) out.push(id);
    }
    return out.sort((a, b) => a - b);
  }

  addSystem(system: System): this {
    this.systems.push(system);
    this.systemsDirty = true;
    return this;
  }

  removeSystem(name: string): boolean {
    const before = this.systems.length;
    const idx = this.systems.findIndex((s) => s.name === name);
    if (idx >= 0) this.systems.splice(idx, 1);
    this.systemsDirty = this.systems.length !== before;
    return idx >= 0;
  }

  get systemNames(): string[] {
    return this.sortedSystems().map((s) => s.name);
  }

  private sortedSystems(): System[] {
    if (this.systemsDirty) {
      const rank = (p?: SystemPhase) => (p ? PHASES.indexOf(p) : 1);
      this.systems.sort((a, b) => {
        const pr = rank(a.phase) - rank(b.phase);
        if (pr !== 0) return pr;
        return (a.order ?? 0) - (b.order ?? 0);
      });
      this.systemsDirty = false;
    }
    return this.systems;
  }

  /** Run one fixed simulation step across all systems, in phase then order. */
  step(dt: number): void {
    const ctx: SystemContext = {
      world: this,
      dt,
      time: this.simulatedTime,
      step: this.stepCount,
    };
    for (const system of this.sortedSystems()) system.update(ctx);
    this.simulatedTime += dt;
    this.stepCount++;
  }

  get stepCounter(): number {
    return this.stepCount;
  }

  get simulatedSeconds(): number {
    return this.simulatedTime;
  }

  private storeFor(name: string): Map<EntityId, unknown> {
    let store = this.stores.get(name);
    if (!store) {
      store = new Map();
      this.stores.set(name, store);
    }
    return store;
  }

  clear(): void {
    for (const store of this.stores.values()) store.clear();
    this.stores.clear();
    this.alive.clear();
    this.removed.length = 0;
    this.nextId = 1;
    this.stepCount = 0;
    this.simulatedTime = 0;
  }
}
